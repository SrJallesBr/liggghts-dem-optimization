#!/usr/bin/env python3
"""Calibracao DEM: DOE Latin Hypercube, resposta quadratica e refinamento."""
from __future__ import annotations
import argparse, concurrent.futures, csv, json, math, random, re, shutil, subprocess, time
from pathlib import Path
from medir_angulos import measure_drawdown, measure_repose
NAMES=("sf11","rf11","rf12","CED11")
FIELDS=["trial","phase",*NAMES,"repose_deg","drawdown_deg","score","repose_left","repose_right","drawdown_left","drawdown_right","status","seconds"]
def launch_dashboard(root):
    """Abre um unico painel no Windows Terminal para acompanhar toda a rodada."""
    if not shutil.which("wt.exe"):
        return
    command=[
        "wt.exe","new-tab","--title","LIGGGHTS - Otimizacao 6x2",
        "wsl.exe","--","python3",str(Path(__file__).resolve()),
        "--root",str(root),"--monitor-dashboard",
    ]
    try:
        subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    except OSError:
        pass

def dashboard(root):
    """Mostra CSV e a ultima linha de cada um dos seis logs ativos em uma aba."""
    csv_path=root/"resultados.csv"
    while True:
        rows=load(csv_path)
        logs=sorted((root/"trials").glob("trial_*/run.log"),key=lambda p:p.stat().st_mtime,reverse=True)[:6]
        print("\033[2J\033[H",end="")
        print("LIGGGHTS - OTIMIZACAO 6x2",time.strftime("%d/%m/%Y %H:%M:%S"))
        print(f"Concluidos: {len(rows)}   Ativos/exibidos: {len(logs)}\n")
        print(f"{'trial':<24} {'fase':<20} {'repouso':>9} {'draftdown':>10} {'score':>10} {'status':<18}")
        print("-"*98)
        for r in rows[-12:]:
            print(f"{r.get('trial',''):<24} {r.get('phase',''):<20} {r.get('repose_deg',''):>9} {r.get('drawdown_deg',''):>10} {r.get('score',''):>10} {r.get('status',''):<18}")
        print("\nULTIMA LINHA DOS LOGS DO LOTE ATUAL")
        print("-"*98)
        for log in reversed(logs):
            try:
                lines=[x.strip() for x in log.read_text(encoding="utf-8",errors="replace").splitlines() if x.strip()]
                last=lines[-1] if lines else "aguardando saida do LIGGGHTS"
            except OSError as exc:
                last=f"log indisponivel: {exc}"
            print(f"{log.parent.name:<32} {last[:64]}")
        print("\nAtualizacao automatica; Ctrl+C fecha somente este painel.",flush=True)
        time.sleep(10)
def load(p):
    if not p.exists(): return []
    with p.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def save(p,r):
    new=not p.exists()
    with p.open("a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS)
        if new:w.writeheader()
        w.writerow({k:r.get(k,"") for k in FIELDS})
def loss(a,b,c):
    t=c["tolerance_deg"];return ((a-c["target_repose_deg"])/t)**2+((b-c["target_drawdown_deg"])/t)**2
def replace(text,n,v):
    out,k=re.subn(rf"(?m)^(variable\s+{re.escape(n)}\s+equal\s+)([^\s#]+)",rf"\g<1>{v:.10g}",text,count=1)
    if k!=1:raise ValueError(f"Parametro {n} nao encontrado de forma unica")
    return out
def trial(num,phase,p,c,root,template,meshes):
    name=f"trial_{num:03d}";d=root/"trials"/name
    if d.exists():
        name=f"trial_{num:03d}_r{int(time.time()*1000)}";d=root/"trials"/name
    d.mkdir(parents=True,exist_ok=False);(d/"post").mkdir()
    try:
        (d/"meshes").symlink_to(meshes,target_is_directory=True)
    except OSError:
        # Google Drive montado em drvfs nao permite symlinks; copia so as quatro meshes.
        shutil.copytree(meshes,d/"meshes")
    text=template.read_text(encoding="utf-8")
    for n,v in p.items():text=replace(text,n,v)
    (d/"in.otimizacao").write_text(text,encoding="utf-8");start=time.time()
    with (d/"run.log").open("w",encoding="utf-8") as f:r=subprocess.run(["mpirun","--bind-to","none","-np",str(c["mpi_processes_per_trial"]),"liggghts","-in","in.otimizacao"],cwd=d,stdout=f,stderr=subprocess.STDOUT)
    out={"trial":name,"phase":phase,**p,"status":"failed","seconds":f"{time.time()-start:.1f}"}
    if r.returncode:return out
    try:
        a=measure_repose(d/"post"/"angulo_repouso.vtk");b=measure_drawdown(d/"post"/"drawdown_final.vtk")
        out.update({"repose_deg":f"{a['mean_deg']:.6f}","drawdown_deg":f"{b['mean_deg']:.6f}","score":f"{loss(a['mean_deg'],b['mean_deg'],c):.8f}","repose_left":f"{a['left']['angle_deg']:.6f}","repose_right":f"{a['right']['angle_deg']:.6f}","drawdown_left":f"{b['left']['angle_deg']:.6f}","drawdown_right":f"{b['right']['angle_deg']:.6f}","status":"ok"})
        (d/"measurement.json").write_text(json.dumps({"repose":a,"drawdown":b},indent=2))
    except Exception as e:out["status"]=f"measurement_failed: {e}"
    return out
def normalize(p,c):return [(p[n]-c["parameters"][n]["min"])/(c["parameters"][n]["max"]-c["parameters"][n]["min"])*2-1 for n in NAMES]
def terms(x):return [1,*x,*[v*v for v in x],*[x[i]*x[j] for i in range(4) for j in range(i+1,4)]]
def gaussian(a,b):
    n=len(b);a=[r[:]+[b[i]] for i,r in enumerate(a)]
    for col in range(n):
        q=max(range(col,n),key=lambda i:abs(a[i][col]))
        if abs(a[q][col])<1e-12:raise ValueError("modelo singular")
        a[col],a[q]=a[q],a[col];z=a[col][col];a[col]=[v/z for v in a[col]]
        for row in range(n):
            if row!=col:
                z=a[row][col];a[row]=[a[row][j]-z*a[col][j] for j in range(n+1)]
    return [a[i][-1] for i in range(n)]
def fit(rows,key,c):
    X=[terms(normalize({n:float(r[n]) for n in NAMES},c)) for r in rows];y=[float(r[key]) for r in rows];m=15
    A=[[sum(x[i]*x[j] for x in X)+(1e-8 if i==j else 0) for j in range(m)] for i in range(m)]
    return gaussian(A,[sum(x[i]*v for x,v in zip(X,y)) for i in range(m)])
def predict(m,p,c):return sum(a*b for a,b in zip(m,terms(normalize(p,c))))
def lhs(count,c,rng):
    cols=[]
    for n in NAMES:
        order=list(range(count));rng.shuffle(order);s=c["parameters"][n]
        cols.append([s["min"]+(i+rng.random())/count*(s["max"]-s["min"]) for i in order])
    return [{n:cols[j][i] for j,n in enumerate(NAMES)} for i in range(count)]
def repeated(p,rows,q):
    for r in list(rows)+q:
        if all(abs(p[n]-float(r[n]))<(0.1 if n=="CED11" else 1e-5) for n in NAMES):return True
    return False
def suggest(rows,c,rng,count):
    ok=[r for r in rows if r.get("status")=="ok"]
    if len(ok)<16:return lhs(count,c,rng)
    ma,mb=fit(ok,"repose_deg",c),fit(ok,"drawdown_deg",c)
    pool=lhs(c["model_candidate_pool"],c,rng);pool.sort(key=lambda p:loss(predict(ma,p,c),predict(mb,p,c),c));out=[]
    for p in pool:
        if not repeated(p,rows,out):out.append(p)
        if len(out)==count:return out
    return out
def report(rows,c,root):
    ok=[r for r in rows if r.get("status")=="ok"]
    train=[r for r in ok if r.get("phase")!="validacao_final"]
    check=[r for r in ok if r.get("phase")=="validacao_final"]
    if len(train)<15 or not check:return
    ma,mb=fit(train,"repose_deg",c),fit(train,"drawdown_deg",c)
    def rmse(k,m):return math.sqrt(sum((predict(m,{n:float(r[n]) for n in NAMES},c)-float(r[k]))**2 for r in check)/len(check))
    best=min(ok,key=lambda r:float(r["score"]));bp={n:float(best[n]) for n in NAMES};sen={}
    for n in NAMES:
        s=c["parameters"][n];h=.01*(s["max"]-s["min"]);hi=bp.copy();lo=bp.copy();hi[n]=min(s["max"],bp[n]+h);lo[n]=max(s["min"],bp[n]-h)
        sen[n]={"repose_deg_por_1pct_faixa":(predict(ma,hi,c)-predict(ma,lo,c))/2,"drawdown_deg_por_1pct_faixa":(predict(mb,hi,c)-predict(mb,lo,c))/2}
    (root/"ANALISE_DOE.json").write_text(json.dumps({"quadratic_terms":15,"training_trials":len(train),"validation_trials":len(check),"validation_rmse_deg":{"repose":rmse("repose_deg",ma),"drawdown":rmse("drawdown_deg",mb)},"best_observed":best,"local_sensitivity":sen},indent=2),encoding="utf-8")
def converged(r,c):return r.get("status")=="ok" and abs(float(r["repose_deg"])-c["target_repose_deg"])<=c["tolerance_deg"] and abs(float(r["drawdown_deg"])-c["target_drawdown_deg"])<=c["tolerance_deg"]
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path.home()/"dem_optimization");ap.add_argument("--monitor-dashboard",action="store_true");args=ap.parse_args();root=args.root.resolve()
    if args.monitor_dashboard:
        dashboard(root);return
    c=json.loads((root/"configuracao_otimizacao.txt").read_text());history=load(root/"resultados.csv");rows=[r for r in history if r.get("status")=="ok"];rng=random.Random(c["random_seed"]);doe=lhs(c["doe_trials"],c,rng);template,meshes=root/"in.repousodraftdown_otimizacao",root/"meshes"
    launch_dashboard(root)
    while len(rows)<c["max_trials"]:
        phase="doe" if len(rows)<c["doe_trials"] else ("refinamento_modelo" if len(rows)<c["doe_trials"]+c["refinement_trials"] else "validacao_final");size=min(c["parallel_trials"],c["max_trials"]-len(rows));batch=[]
        source=doe if phase=="doe" else (suggest(rows,c,rng,size) if phase=="refinamento_modelo" else lhs(size,c,rng))
        while len(batch)<size:
            i=len(rows)+len(batch);p=source[i] if phase=="doe" else source[len(batch)]
            if not repeated(p,rows,batch):batch.append(p)
            else:source=lhs(max(20,size*8),c,rng)
        first=len(history)+1;print(f"Fase {phase}: lote {first}..{first+len(batch)-1}",flush=True);batch_results=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=size) as ex:
            fs=[ex.submit(trial,first+i,phase,p,c,root,template,meshes) for i,p in enumerate(batch)]
            for f in concurrent.futures.as_completed(fs):
                r=f.result();save(root/"resultados.csv",r);r={k:str(v) for k,v in r.items()};history.append(r);batch_results.append(r);print(json.dumps(r),flush=True)
                if r.get("status")=="ok":rows.append(r)
        failed=[r for r in batch_results if r.get("status")!="ok"]
        if failed:
            print(f"LOTE INTERROMPIDO: {len(failed)} trial(s) falharam; corrija o erro e reinicie. Resultados invalidos nao contam para as 48 simulacoes.",flush=True);return
        report(rows,c,root);wins=[r for r in rows if converged(r,c)]
        if wins:
            best=min(wins,key=lambda r:float(r["score"]));(root/"MELHOR_RESULTADO.json").write_text(json.dumps(best,indent=2));print("CONVERGIU: "+json.dumps(best),flush=True);return
    ok=[r for r in rows if r.get("status")=="ok"]
    if ok:(root/"MELHOR_RESULTADO.json").write_text(json.dumps(min(ok,key=lambda r:float(r["score"])),indent=2))
if __name__=="__main__":main()

