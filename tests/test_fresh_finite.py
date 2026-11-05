"""Offline integration tests: no production solver or input execution."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from optimizer.calibration_controller import CalibrationController, MeasurementRejected
from optimizer.adaptive_refinement import run_adaptive_refinement
from runtime import run_fresh_finite as fresh

class FreshFiniteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # Fake solver has no real provenance receipt. Stub ONLY its admission
        # boundary, never production code; a separate test checks the real gate.
        admission = patch('optimizer.protocol_eligibility._admitted', return_value=True)
        admission.start()
        self.addCleanup(admission.stop)
        self.root = Path(self.tmp.name)/'fresh'
        shutil.copytree(ROOT, self.root, ignore=shutil.ignore_patterns('run', '__pycache__', '*.pyc', 'tests'))

    def controller(self):
        return CalibrationController(self.root, run_root=self.root/'run')

    def result(self, c, tid, p, phase):
        r = dict(p, trial_id=tid, phase=phase, status='complete')
        for k in c.config['tests']:
            r.update({k+'_deg':20., k+'_response_status':'measured', k+'_measurement_valid':True})
        c._append_results(r)
        return r

    def test_budget_and_initial_design(self):
        c = self.controller()
        self.assertEqual(c.config['maximum_doe_attempts'],420)
        self.assertEqual(c.config['minimum_doe_complete'],84)
        self.assertEqual(c.config['parallel_trials'],20)
        self.assertEqual(len(c.initial_doe_points()),84)
        self.assertEqual(len({c._point_key(p) for p in c.initial_doe_points()}),84)

    def test_retry_to_84_preserves_extra_inflight_and_restart(self):
        c=self.controller()
        def fake(tid,p,phase='doe',seeds=None):
            if tid < 11:
                raise MeasurementRejected('synthetic unmeasurable geometry')
            return self.result(c,tid,p,phase)
        with patch.object(c,'execute_trial',side_effect=fake):
            rows=c.run_doe()
        records=c._restore_async_doe_state()
        self.assertGreaterEqual(len(records),94)
        self.assertLessEqual(len(records),113)
        self.assertEqual(sum(c._valid_responses(r) for r in rows),len(records)-10)
        self.assertEqual(len({c._point_key(r['parameters']) for r in records.values()}),len(records))
        self.assertTrue(c._doe_is_ready(rows))
        fresh.verify_isolation(self.root)
        with patch.object(c,'execute_trial',side_effect=AssertionError('resume launched duplicate')):
            c.run_doe()
        self.assertEqual(len(c._restore_async_doe_state()),len(records))

    def test_doe_cap420_never_refines_insufficient_data(self):
        c=self.controller()
        with patch.object(c,'execute_trial',side_effect=MeasurementRejected('synthetic rejection')):
            rows=c.run_doe()
        self.assertEqual(len(c._restore_async_doe_state()),420)
        self.assertFalse(c._doe_is_ready(rows))
        with patch.object(fresh,'run_adaptive_refinement',side_effect=AssertionError('refined without valid DOE')):
            summary=fresh.run(c)
        self.assertEqual(summary['status'],'doe_insufficient')

    def test_global1000_limits_refinement_after420_attempts(self):
        from optimizer.adaptive_refinement import _configured_budget_limit
        c=self.controller()
        records={i:{'phase':'doe'} for i in range(420)}
        with patch.object(c,'_restore_async_doe_state',return_value=records):
            self.assertEqual(_configured_budget_limit(c),580)

    def test_technical_error_fail_closed(self):
        c=self.controller()
        with patch.object(c,'execute_trial',side_effect=RuntimeError('synthetic infrastructure error')):
            c.run_doe()
        self.assertLessEqual(len(c._restore_async_doe_state()),20)

    def test_unowned_history_rejected(self):
        (self.root/'run').mkdir()
        (self.root/'run/results.csv').write_text('trial_id,phase\n593,doe\n')
        with self.assertRaisesRegex(RuntimeError,'unowned'):
            fresh.verify_isolation(self.root)

    def test_default_launch_denied(self):
        c=self.controller()
        with self.assertRaisesRegex(RuntimeError,'HOLD'):
            fresh.require_fresh_authorization(self.root,c.config)

    def test_refinement_20_workers_fake_solver_and_restart(self):
        c=self.controller()
        with patch.object(c,'execute_trial',side_effect=lambda tid,p,phase='doe',seeds=None:self.result(c,tid,p,phase)):
            c.run_doe()
        def propose(c,rows,attempted,proposal_index=0):
            import random
            rng=random.Random(900000+proposal_index)
            return {n:s['min']+rng.random()*(s['max']-s['min']) for n,s in c.config['parameters'].items()}, {'fake':True}
        with patch.object(c,'execute_trial',side_effect=lambda tid,p,phase='refinement',seeds=None:self.result(c,tid,p,phase)):
            summary=run_adaptive_refinement(c,workers=20,budget=748,proposer=propose)
        self.assertEqual(summary['attempts'],748)
        self.assertEqual(summary['status'],'budget_exhausted')
        self.assertEqual(summary['workers'],20)
        fresh.verify_isolation(self.root)
        with patch.object(c,'execute_trial',side_effect=AssertionError('duplicate refinement')):
            again=run_adaptive_refinement(c,workers=20,budget=748,proposer=propose)
        self.assertEqual(again['attempts'],748)

if __name__=='__main__': unittest.main(verbosity=2)
