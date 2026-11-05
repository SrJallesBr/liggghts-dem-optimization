"""Manifest-correlated, bounded audit notifications. No solver imports or starts.

An interrupted/ambiguous send is kept UNKNOWN and never retried automatically.
Accepted means transport acceptance, not that the recipient viewed the image.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


def scan(root):
    root=Path(root).resolve();events=[]
    if not root.is_dir(): return events
    for path in sorted(root.rglob('Audit_Manifest.json')):
        if not path.resolve().is_relative_to(root): raise ValueError('manifest escapes root')
        payload=path.read_bytes();data=json.loads(payload)
        for artifact in data['artifacts']:
            image=(path.parent/artifact['path']).resolve()
            if not image.is_relative_to(path.parent.resolve()): raise ValueError('artifact escapes manifest directory')
            if hashlib.sha256(image.read_bytes()).hexdigest()!=artifact['sha256']:
                raise ValueError('artifact hash mismatch: '+image.name)
            if image.suffix.lower()!='.png':continue
            key=hashlib.sha256((path.relative_to(root).as_posix()+hashlib.sha256(payload).hexdigest()+artifact['path']+artifact['sha256']).encode()).hexdigest()
            trial=Path(data['source_path']).parents[2].name
            caption=f"Auditoria DEM | {trial} | {data['kind']} | {data['status']}\n{data.get('reason') or 'Medido; não significa calibração validada.'}\nDados salvos; sem iniciar DEM. Equivalência ANSYS não validada."
            events.append({'id':key,'manifest':str(path),'source_sha256':data['source_sha256'],
                           'body':caption+'\nMEDIA:'+image.as_posix(),'image':str(image)})
    return events


def save_state(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    os.replace(temporary,path)


def dispatch(root,state,target,transport,limit=2,baseline=False):
    state=Path(state);state.parent.mkdir(parents=True,exist_ok=True)
    lock=state.with_suffix(state.suffix+'.lock')
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:return {'accepted':0,'unknown':0,'locked':True}
    os.close(fd)
    try:
        current=json.loads(state.read_text('utf8')) if state.exists() else {'schema':1,'events':{}}
        output={'accepted':0,'unknown':0,'baseline':0}
        attempts=0
        for event in scan(root):
            key=target+'|'+str(Path(root).resolve())+'|'+event['id']
            if key in current['events']:continue
            if not baseline and attempts>=limit:break
            record={'status':'baseline' if baseline else 'in_flight','image':event['image'],
                    'at_utc':datetime.now(timezone.utc).isoformat()}
            current['events'][key]=record;save_state(state,current)
            if baseline:
                output['baseline']+=1;continue
            attempts+=1
            # If the owner crashes during transport, in_flight is also skipped
            # on restart: it may already have reached the recipient.
            try:
                receipt=transport(target,event['body'])
                if not receipt.get('success') or receipt.get('error'):
                    raise RuntimeError('transport acceptance not confirmed')
                record.update(status='accepted',receipt=receipt)
                output['accepted']+=1
            except Exception as exc:
                record.update(status='unknown',error_type=type(exc).__name__,review_required=True)
                output['unknown']+=1
            save_state(state,current)
        return output
    finally:lock.unlink(missing_ok=True)


def hermes_transport(python,hermes_root,target,body):
    p=subprocess.run([str(python),'-E','-X','utf8','-m','hermes_cli.main','send','--to',target,'--json',body],
                     cwd=hermes_root,capture_output=True,text=True,encoding='utf8',timeout=90)
    result=json.loads(p.stdout)
    if p.returncode!=0 or not result.get('success') or result.get('skipped') or result.get('error'):
        raise RuntimeError('Hermes send did not confirm acceptance')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path);p.add_argument('--state',required=True,type=Path)
    p.add_argument('--target',required=True);p.add_argument('--limit',type=int,default=2)
    action=p.add_mutually_exclusive_group();action.add_argument('--send',action='store_true');action.add_argument('--baseline',action='store_true')
    p.add_argument('--hermes-python',type=Path);p.add_argument('--hermes-root',type=Path)
    a=p.parse_args(argv)
    if a.limit<1:p.error('limit must be positive')
    if not a.send and not a.baseline:
        print(json.dumps(scan(a.root),ensure_ascii=False));return 0
    if a.send and not (a.hermes_python and a.hermes_root):p.error('--send requires explicit Hermes Python/root')
    result=dispatch(a.root,a.state,a.target,
        lambda target,body:hermes_transport(a.hermes_python,a.hermes_root,target,body),a.limit,a.baseline)
    if result.get('accepted') or result.get('unknown') or a.baseline:
        print(json.dumps(result,ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
