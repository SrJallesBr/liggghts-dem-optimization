#!/usr/bin/env python3
"""Read-only campaign dashboard. No numpy, solver import, locks or writes.
Reads authoritative status events, not the scheduler's stale status_execucao.csv.
Running in the journal is not proof that a solver is alive.
"""
import argparse
import csv
import io
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

TERMINAL = {'complete', 'failed', 'rejected', 'geometry_unmeasurable'}


def complete_text(path):
    try:
        text = path.read_text(encoding='utf-8-sig', errors='replace')
    except FileNotFoundError:
        return ''
    # A producer may be halfway through its last append. Try again next refresh.
    return text[:text.rfind('\n') + 1] if text else ''


def snapshot(root, now=None):
    root = Path(root)
    now = time.time() if now is None else now
    latest = {}
    statuses = root/'status.csv'
    for row in csv.DictReader(io.StringIO(complete_text(statuses))):
        try:
            tid = int(row.get('trial_id', ''))
        except (ValueError, TypeError):
            continue
        if not row.get('status') or row.get('stage') is None:
            continue
        latest[tid] = dict(row, trial_id=tid)
    # Journals distinguish rejection from a missing or completed measurement.
    for name in ('doe_schedule.jsonl', 'adaptive_proposals.jsonl'):
        for line in complete_text(root/name).splitlines():
            try:
                event = json.loads(line)
                tid = int(event['trial_id'])
            except (ValueError, KeyError, TypeError):
                continue
            phase = 'doe' if name == 'doe_schedule.jsonl' else 'refinement'
            if event.get('event') == 'reserved':
                latest.setdefault(tid, {'trial_id':tid, 'phase':phase, 'status':'reserved', 'stage':'queue'})
            status = {'completed':'complete', 'rejected':'rejected', 'failed':'failed'}.get(event.get('event'))
            if status:
                latest.setdefault(tid, {'trial_id':tid, 'phase':phase})['status'] = status
    measurements = {phase:{kind:0 for kind in ('discharge','repose','rolling')}
                    for phase in ('doe','refinement','validation')}
    results = {}
    for row in csv.DictReader(io.StringIO(complete_text(root/'results.csv'))):
        try:
            results[int(row['trial_id'])] = row
        except (KeyError, TypeError, ValueError):
            continue
    accepted = 0
    for tid, row in results.items():
        phase = row.get('phase', '?')
        # Results are persisted after measurement; do not relabel a journal failure.
        latest.setdefault(tid, {'trial_id':tid, 'phase':phase, 'status':row.get('status') or 'unknown'})
        accepted += row.get('accepted', '').strip().lower() == 'true'
        if phase not in measurements:
            continue
        for kind in measurements[phase]:
            try:
                angle = float(row.get(kind+'_deg', ''))
            except (TypeError, ValueError):
                continue
            if row.get(kind+'_measurement_valid', '').strip().lower() == 'true' and math.isfinite(angle):
                measurements[phase][kind] += 1
    counts = Counter(row['status'] for row in latest.values())
    active = []
    for tid, row in sorted(latest.items()):
        if row['status'] in TERMINAL:
            continue
        row = dict(row)
        row['log_age'] = None
        row['step'] = '?'
        folder = root/f'trial_{tid:04d}'
        # Report the latest log; timestamps indicate activity, not solver liveness.
        logs = [p for p in (folder/n for n in ('discharge.log','repose.log','rolling_fill.log','rolling.log')) if p.is_file()]
        if logs:
            log = max(logs, key=lambda p:p.stat().st_mtime)
            row['log_age'] = max(0, int(now-log.stat().st_mtime))
            row['log'] = log.name
            with log.open('rb') as handle:
                handle.seek(max(0, log.stat().st_size-65536))
                lines = handle.read().decode('utf-8', errors='replace').splitlines()
            for line in reversed(lines):
                fields = line.split()
                if len(fields) >= 2 and fields[0].isdigit():
                    try:
                        float(fields[1])
                    except ValueError:
                        continue
                    row['step'] = fields[0]
                    break
        active.append(row)
    try:
        age = max(0, int(now-statuses.stat().st_mtime))
    except FileNotFoundError:
        age = None
    return {'active':active, 'counts':dict(counts), 'status_age':age, 'now':now,
            'measurements':measurements, 'accepted':accepted}


def render(data):
    c = data['counts']
    lines = ['=== PAINEL DEM (leitura independente) ===',
             datetime.fromtimestamp(data['now']).strftime('%Y-%m-%d %H:%M:%S'),
             f"Concluidos registrados: {c.get('complete',0)} | Rejeitados: {c.get('rejected',0)+c.get('geometry_unmeasurable',0)} | Falhas: {c.get('failed',0)}",
             f"Alvos aceitos registrados: {data['accepted']} (nao e validacao independente)",
             'Concluido nao significa ponto valido nem alvo atingido.']
    for phase, values in data['measurements'].items():
        lines.append(f"Validos sinalizados [{phase}]: descarga={values['discharge']} repouso={values['repose']} rolamento={values['rolling']}")
    lines += [f"Ultimo evento status.csv: {data['status_age']}s atras (idade nao indica pausa)",
             f"{'TRIAL':<12} {'FASE':<13} {'ETAPA':<18} {'PASSO LOG':<12} {'LOG IDADE':<12}"]
    for r in data['active']:
        age = '?' if r['log_age'] is None else str(r['log_age'])+'s'
        lines.append(f"{r['trial_id']:<12} {r.get('phase','?'):<13} {r.get('stage','?'):<18} {r['step']:<12} {age:<12}")
    lines.append('Ativos = pendentes registrados; nao confirma processos vivos. Passo vem da cauda do log, nao e tempo fisico.')
    if not data['active']:
        lines.append('Nenhum pendente registrado. Verifique finite_summary.json/aba do controlador; nao equivale a convergencia.')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=Path(__file__).resolve().parents[1]/'run')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=10)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    if not args.run_root.is_dir():
        parser.error('run-root directory not found')
    try:
        while True:
            try:
                text = render(snapshot(args.run_root))
            except (OSError, csv.Error) as exc:
                text = 'Falha ao ler monitor (campanha nao alterada): '+str(exc)
                if args.once:
                    print(text, file=sys.stderr)
                    return 1
            if sys.stdout.isatty() and not args.once:
                print('\033[2J\033[H', end='')
            print(text, flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0

if __name__ == '__main__':
    raise SystemExit(main())
