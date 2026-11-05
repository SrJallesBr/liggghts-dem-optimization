"""Offline admission/integration tests. Synthetic execution is not physical validation."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from optimizer.calibration_controller import CalibrationController, MeasurementRejected
from optimizer.fresh_authorization import require_fresh_authorization, SCOPE
from optimizer.repose_ansys import require_scientific_protocol_compatibility
from runtime import run_fresh_finite as fresh

class FreshAdmissionTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.root=Path(self.tmp.name)/'package'
  shutil.copytree(ROOT,self.root,ignore=shutil.ignore_patterns('run','__pycache__','*.pyc'))
  self.c=CalibrationController(self.root,run_root=self.root/'run')
 def auth(self):return patch.dict(os.environ,{'DEM_FRESH_FINITE_AUTHORIZATION_ID':SCOPE})
 def test_prepare_first_fresh_trial_uses_current_templates(self):
  point=self.c.initial_doe_points()[0]
  workspace=self.c.prepare_trial(1,point)
  self.assertTrue((workspace/'Ang Repouso/initial_particles.data').is_file())
  deck=(workspace/'Ang Repouso/in.repouso').read_text()
  self.assertIn('ANSYS_REPOSE_10S_ONE_SHOT',deck)
  self.assertIn('repouso_10s.vtk',deck)
  self.assertTrue((workspace/'Ang Rolamento/in.rolamento').is_file())
 def test_config_allows_explicit_start(self):
  self.assertTrue(self.c.config['simulation_launch_enabled'])
 def test_actual_surrogate_proposal_on_synthetic_doe(self):
  c=self.c
  def execute(tid,p,phase='doe',seeds=None):
   r=dict(p,trial_id=tid,phase=phase,status='complete',workspace=str(c.run_root/f'trial_{tid:04d}'))
   z=sum((p[n]-s['min'])/(s['max']-s['min']) for n,s in c.config['parameters'].items())
   for k in c.config['tests']:r.update({k+'_deg':20.+z,k+'_response_status':'measured',k+'_measurement_valid':True})
   c._append_results(r);return r
  with patch('optimizer.protocol_eligibility._admitted',return_value=True),patch.object(c,'execute_trial',side_effect=execute):
   c.run_doe()
   summary=fresh.run_adaptive_refinement(c,workers=20,budget=1)
  self.assertEqual(summary['attempts'],1)
  self.assertEqual(summary['status'],'budget_exhausted')
 def test_cli_check_with_synthetic_capacity_never_executes(self):
  with patch.object(fresh,'ROOT',self.root),patch.object(sys,'argv',['run_fresh_finite.py','--check','--solver',sys.executable]),patch('optimizer.calibration_controller.os.cpu_count',return_value=20),patch.object(fresh,'run',side_effect=AssertionError('check started a campaign')),patch('optimizer.calibration_controller.subprocess.Popen',side_effect=AssertionError('check started a solver')):
   fresh.main()
 def test_science_gate_accepts_pinned_fresh_copy_only(self):
  with self.auth(): require_scientific_protocol_compatibility(self.root,self.c.config)
 def test_changed_factor_configuration_rejected(self):
  cfg=dict(self.c.config);cfg['minimum_doe_complete']=1
  with self.auth(),self.assertRaisesRegex(RuntimeError,'configuration|budget'):
   require_fresh_authorization(self.root,cfg)
 def test_tampered_implementation_rejected(self):
  path=self.root/'optimizer/protocol_eligibility.py';path.write_text(path.read_text()+'\n# altered\n')
  with self.auth(),self.assertRaisesRegex(RuntimeError,'artifact'):
   require_fresh_authorization(self.root,self.c.config)
 def test_unknown_history_or_workspace_rejected(self):
  (self.root/'run').mkdir(exist_ok=True)
  (self.root/'run/trial_0001').mkdir()
  with self.assertRaisesRegex(RuntimeError,'unowned'):fresh.verify_isolation(self.root)
 def test_real_subprocess_fixture_through_actual_launch_gates(self):
  # ONLY a Python marker program; no DEM binary is invoked, no synthetic measurement admitted.
  self.c.solver_command=[sys.executable,'-c','print("OFFLINE_FIXTURE_ONLY")']
  w=self.root/'run/trial_0001';(w/'Ang Repouso').mkdir(parents=True)
  (w/'Ang Repouso/in.repouso').write_text('# offline fixture, not solver input\n')
  with self.auth(): self.c._run_input(w,'Ang Repouso','in.repouso','repose')
  self.assertIn('OFFLINE_FIXTURE_ONLY',(w/'repose.log').read_text())
 def test_controller_without_opt_in_cannot_launch_even_with_enabled_flag(self):
  self.c.solver_command=[sys.executable,'-c','raise SystemExit(99)']
  with patch.dict(os.environ,{},clear=True),self.assertRaisesRegex(RuntimeError,'authorization'):
   self.c._run_input(self.root/'run/trial_0001','Ang Repouso','in.repouso','repose')
 def test_no_real_measurement_receipt_cannot_enter_fit(self):
  from optimizer.protocol_eligibility import interpret_for_controller
  row=dict(self.c.initial_doe_points()[0],trial_id=1,status='complete',phase='doe',workspace=str(self.root/'run/trial_0001'),repose_deg=27.,repose_response_status='measured',repose_measurement_valid=True)
  self.assertFalse(interpret_for_controller(self.c,row,'repose')['regression_included'])
 def test_main_hands_off_doe_to_refinement(self):
  # Exercise real schedulers with injected responses/proposer, not a DEM solver.
  c=self.c
  def execute(tid,p,phase='doe',seeds=None):
   row=dict(p,trial_id=tid,phase=phase,status='complete')
   for k in c.config['tests']:row.update({k+'_deg':20.,k+'_response_status':'measured',k+'_measurement_valid':True})
   c._append_results(row);return row
  def propose(c,rows,attempted,proposal_index=0):
   import random
   rng=random.Random(999000+proposal_index)
   return {n:s['min']+rng.random()*(s['max']-s['min']) for n,s in c.config['parameters'].items()},{'synthetic_fixture':True}
  from optimizer.adaptive_refinement import run_adaptive_refinement as actual
  def refine(*args,**kwargs):return actual(*args,**kwargs,proposer=propose)
  with patch('optimizer.protocol_eligibility._admitted',return_value=True),patch.object(c,'execute_trial',side_effect=execute),patch.object(fresh,'run_adaptive_refinement',side_effect=refine):
   summary=fresh.run(c)
  self.assertTrue(summary['refinement_started']);self.assertEqual(summary['refinement']['attempts'],748)
  self.assertLessEqual(len(c._restore_async_doe_state()),1000)
  # Full occupancy may leave up to 19 extra valid DOE in flight at handoff.
  self.assertGreaterEqual(sum(e['phase']=='doe' for e in c._restore_async_doe_state().values()),84)
  self.assertLessEqual(sum(e['phase']=='doe' for e in c._restore_async_doe_state().values()),103)
  fresh.verify_isolation(self.root)

if __name__=='__main__':unittest.main(verbosity=2)
