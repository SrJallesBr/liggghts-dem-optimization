"""Offline, event-controlled scheduler regression tests; no DEM solver is run."""
import json
import random
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from optimizer.calibration_controller import CalibrationController, MeasurementRejected
from optimizer.adaptive_refinement import run_adaptive_refinement
from runtime import run_fresh_finite as fresh


class ParallelHandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'package'
        shutil.copytree(ROOT, self.root, ignore=shutil.ignore_patterns('run', '.git', '__pycache__', '*.pyc'))
        self.c = CalibrationController(self.root, run_root=self.root/'run')
        self.admit = patch('optimizer.protocol_eligibility._admitted', return_value=True)
        self.admit.start()
        self.addCleanup(self.admit.stop)
        self.errors = []
        self.output = []

    def row(self, tid, point, phase):
        row = dict(point, trial_id=tid, phase=phase, status='complete')
        for k in self.c.config['tests']:
            row.update({k+'_deg': 20., k+'_response_status': 'measured', k+'_measurement_valid': True})
        return row

    def seed(self, count=83):
        for tid, point in enumerate(self.c.initial_doe_points()[:count], 1):
            self.c._append_doe_event(dict(event='reserved', trial_id=tid, parameters=point))
            self.c._append_results(self.row(tid, point, 'doe'))
            self.c._append_doe_event(dict(event='completed', trial_id=tid))

    def launch(self, fn):
        def run():
            try:
                self.output.append(fn())
            except BaseException as exc:
                self.errors.append(exc)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread

    def propose(self, c, rows, attempted, proposal_index=0):
        rng = random.Random(991000+proposal_index)
        return {n:s['min']+rng.random()*(s['max']-s['min']) for n,s in c.config['parameters'].items()}, {'offline_fixture': True}

    def test_missing_one_fills_twenty_and_replenishes_a_rejection(self):
        self.seed()
        gate, full, refilled = threading.Event(), threading.Event(), threading.Event()
        lock = threading.Lock()
        started = []
        def execute(tid, point, phase, seeds):
            with lock:
                started.append(tid)
                if len(started) == 20:
                    full.set()
                if len(started) == 21:
                    refilled.set()
            if tid == 84:
                if not full.wait(5):
                    raise AssertionError('scheduler did not fill all 20 slots with only one valid DOE missing')
                raise MeasurementRejected('offline fixture: invalid measurement')
            if not gate.wait(10):
                raise AssertionError('fixture gate timed out')
            row = self.row(tid, point, phase)
            self.c._append_results(row)
            return row
        with patch.object(self.c, 'execute_trial', side_effect=execute):
            thread = self.launch(self.c.run_doe)
            try:
                self.assertTrue(full.wait(4), 'missing-one tail must occupy 20 workers')
                self.assertTrue(refilled.wait(4), 'a rejected case must be replaced while the other 19 run')
            finally:
                gate.set()
                thread.join(15)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.errors, [])
        self.assertEqual(len(started), 21)
        self.assertEqual(len(self.c._restore_async_doe_state()), 104)

    def test_refinement_overlaps_doe_and_ingests_late_results(self):
        self.seed()
        gate, full, first_refine, late_seen = (threading.Event() for _ in range(4))
        lock = threading.Lock()
        started, observations = [], []
        active = peak = 0
        def execute(tid, point, phase, seeds):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                started.append((tid, phase))
                if sum(p == 'doe' for _, p in started) == 20:
                    full.set()
            try:
                if phase == 'doe':
                    self.assertTrue((full if tid == 84 else gate).wait(10))
                elif not first_refine.is_set():
                    first_refine.set()
                    self.assertTrue(gate.wait(10))
                row = self.row(tid, point, phase)
                self.c._append_results(row)
                return row
            finally:
                with lock:
                    active -= 1
        def propose(c, rows, attempted, proposal_index=0):
            self.assertEqual(len(rows), len({int(r['trial_id']) for r in rows}), 'duplicate model-training rows')
            observed = [int(r['trial_id']) for r in rows if r['phase'] == 'doe']
            observations.append(observed)
            if len(observed) > 84:
                late_seen.set()
            return self.propose(c, rows, attempted, proposal_index)
        def refine(*args, **kwargs):
            # Small synthetic budget; production limits are checked separately.
            kwargs['budget'] = 22
            return run_adaptive_refinement(*args, **kwargs, proposer=propose)
        with patch.object(self.c, 'execute_trial', side_effect=execute), patch.object(fresh, 'run_adaptive_refinement', side_effect=refine):
            thread = self.launch(lambda: fresh.run(self.c))
            try:
                self.assertTrue(first_refine.wait(6), 'refinement must begin before the 19 slow DOE finish')
                self.assertFalse(gate.is_set())
                self.assertEqual(sum(p == 'doe' for _, p in started), 20)
            finally:
                gate.set()
                thread.join(30)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.errors, [])
        self.assertTrue(late_seen.is_set(), observations)
        self.assertLessEqual(peak, 20)
        self.assertEqual(sum(p == 'doe' for _, p in started), 20)
        self.assertEqual(self.output[0]['refinement']['attempts'], 22)
        records = self.c._restore_async_doe_state()
        self.assertEqual(len(records), 125)
        self.assertTrue(all(r['terminal'] for r in records.values()))
        fresh.verify_isolation(self.root)
        # Clean resume: no rerun, no duplicate IDs, same authorized budget.
        with patch.object(self.c, 'execute_trial', side_effect=AssertionError('resume reexecuted a trial')):
            resumed = run_adaptive_refinement(self.c, workers=20, budget=22, proposer=self.propose)
        self.assertEqual(resumed['attempts'], 22)

    def test_global_cap_counts_all_doe_reservations_including_live(self):
        from optimizer.adaptive_refinement import _configured_budget_limit
        records = {tid:dict(phase='doe') for tid in range(1, 421)}
        self.assertEqual(_configured_budget_limit(self.c, doe_records=records), 580)
        records = {tid:dict(phase='doe') for tid in range(1, 104)}
        self.assertEqual(_configured_budget_limit(self.c, doe_records=records), 748)

    def test_attempt_cap_still_stops_replenishment(self):
        self.seed()
        self.c.config['maximum_doe_attempts'] = 85
        started = []
        def execute(tid, point, phase, seeds):
            started.append(tid)
            raise MeasurementRejected('offline invalid-angle fixture')
        with patch.object(self.c, 'execute_trial', side_effect=execute):
            rows = self.c.run_doe()
        self.assertEqual(started, [84, 85])
        self.assertEqual(len(rows), 83)
        self.assertEqual(len(self.c._restore_async_doe_state()), 85)

    def test_dynamic_global_capacity_limits_actual_mixed_dispatch(self):
        self.seed()
        self.c.config['campaign']['max_trials'] = 105
        full, refinements_done, gate = (threading.Event() for _ in range(3))
        lock = threading.Lock()
        started = []
        def execute(tid, point, phase, seeds):
            with lock:
                started.append((tid, phase))
                if sum(p == 'doe' for _, p in started) == 20:
                    full.set()
            if phase == 'doe':
                self.assertTrue((full if tid == 84 else gate).wait(10))
            row = self.row(tid, point, phase)
            self.c._append_results(row)
            if phase == 'refinement' and sum(p == 'refinement' for _, p in started) == 2:
                refinements_done.set()
            return row
        def refine(*args, **kwargs):
            return run_adaptive_refinement(*args, **kwargs, proposer=self.propose)
        with patch.object(self.c, 'execute_trial', side_effect=execute), patch.object(fresh, 'run_adaptive_refinement', side_effect=refine):
            thread = self.launch(lambda: fresh.run(self.c))
            try:
                self.assertTrue(refinements_done.wait(6), 'remaining global capacity must be exactly two')
            finally:
                gate.set()
                thread.join(20)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.errors, [])
        self.assertEqual(self.output[0]['refinement']['attempts'], 2)
        self.assertEqual(len(self.c._restore_async_doe_state()), 105)
        self.assertEqual(max(tid for tid, _ in started), 105)

    def test_late_technical_error_drains_both_phases_without_cancel(self):
        self.seed()
        full, first_refine, error_ingested, gate = (threading.Event() for _ in range(4))
        lock = threading.Lock()
        started = []
        append = self.c._append_doe_event
        def event(e):
            append(e)
            if e['event'] == 'failed':
                error_ingested.set()
        def execute(tid, point, phase, seeds):
            with lock:
                started.append((tid, phase))
                if sum(p == 'doe' for _, p in started) == 20:
                    full.set()
            if phase == 'doe' and tid == 84:
                self.assertTrue(full.wait(10))
            elif phase == 'doe' and tid == 85:
                self.assertTrue(first_refine.wait(10))
                raise RuntimeError('offline technical-error fixture')
            else:
                if phase == 'refinement':
                    first_refine.set()
                self.assertTrue(gate.wait(10))
            row = self.row(tid, point, phase)
            self.c._append_results(row)
            return row
        def refine(*args, **kwargs):
            kwargs['budget'] = 22
            return run_adaptive_refinement(*args, **kwargs, proposer=self.propose)
        with patch.object(self.c, 'execute_trial', side_effect=execute), patch.object(self.c, '_append_doe_event', side_effect=event), patch.object(fresh, 'run_adaptive_refinement', side_effect=refine):
            thread = self.launch(lambda: fresh.run(self.c))
            try:
                self.assertTrue(error_ingested.wait(6), 'late DOE error not ingested while refinement runs')
            finally:
                gate.set()
                thread.join(20)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(self.errors), 1)
        self.assertIn('technical error in in-flight DOE', str(self.errors[0]))
        self.assertEqual(sum(p == 'refinement' for _, p in started), 1)
        records = self.c._restore_async_doe_state()
        self.assertTrue(all(r['terminal'] for r in records.values()))
        report = json.loads((self.c.run_root/'adaptive_summary.json').read_text())
        self.assertEqual(report['status'], 'error')
        fresh.verify_isolation(self.root)

    def test_candidate_drains_remaining_doe_before_return(self):
        self.seed()
        full, candidate, gate = (threading.Event() for _ in range(3))
        lock = threading.Lock()
        started = []
        def execute(tid, point, phase, seeds):
            with lock:
                started.append((tid, phase))
                if sum(p == 'doe' for _, p in started) == 20:
                    full.set()
            if phase == 'doe':
                self.assertTrue((full if tid == 84 else gate).wait(10))
            row = self.row(tid, point, phase)
            if phase == 'refinement':
                for kind in self.c.config['tests']:
                    row[kind+'_deg'] = self.c.config['target_'+kind+'_deg']
            self.c._append_results(row)
            if phase == 'refinement':
                candidate.set()
            return row
        def refine(*args, **kwargs):
            kwargs['budget'] = 22
            return run_adaptive_refinement(*args, **kwargs, proposer=self.propose)
        with patch.object(self.c, 'execute_trial', side_effect=execute), patch.object(fresh, 'run_adaptive_refinement', side_effect=refine):
            thread = self.launch(lambda: fresh.run(self.c))
            try:
                self.assertTrue(candidate.wait(6))
                self.assertTrue(thread.is_alive(), 'candidate must not abandon live DOE')
            finally:
                gate.set()
                thread.join(20)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.errors, [])
        self.assertEqual(self.output[0]['status'], 'candidate_found')
        self.assertEqual(sum(p == 'refinement' for _, p in started), 1)
        self.assertTrue(all(r['terminal'] for r in self.c._restore_async_doe_state().values()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
