import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class MonitorTests(unittest.TestCase):
    def load(self):
        spec = importlib.util.spec_from_file_location('monitor_campaign', ROOT/'runtime/monitor_campaign.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_stale_live_csv_is_not_authoritative(self):
        m = self.load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'status_execucao.csv').write_text('trial,phase,status\ntrial_0021,doe,rolling\n')
            (root/'status.csv').write_text('trial_id,phase,status,stage,started_at,finished_at\n21,doe,complete,measure,2026-11-05T13:00:00+00:00,2026-11-05T13:10:00+00:00\n84,doe,running,discharge,2026-11-05T14:00:00+00:00,\n')
            (root/'doe_schedule.jsonl').write_text('{"event":"rejected","trial_id":77}\n{"event":')
            before = {p.name:p.read_bytes() for p in root.iterdir()}
            snapshot = m.snapshot(root)
            self.assertEqual([r['trial_id'] for r in snapshot['active']], [84])
            self.assertEqual(snapshot['counts']['complete'], 1)
            self.assertEqual(snapshot['counts']['rejected'], 1)
            self.assertEqual(before, {p.name:p.read_bytes() for p in root.iterdir()})

    def test_running_does_not_claim_solver_alive(self):
        m = self.load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'status.csv').write_text('trial_id,phase,status,stage\n1,doe,running,rolling\n1,doe,failed,failed\n2,refinement,running,repose\n')
            data = m.snapshot(root)
            self.assertEqual(data['counts']['failed'], 1)
            self.assertEqual(len(data['active']), 1)
            text = m.render(data)
            self.assertIn('registrados', text)
            self.assertIn('nao confirma', text)

    def test_missing_files_and_partial_csv(self):
        m = self.load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(m.snapshot(root)['active'], [])
            (root/'status.csv').write_text('trial_id,phase,status,stage\n1,doe,running,rolling\n2,doe,run')
            self.assertEqual(len(m.snapshot(root)['active']), 1)

    def test_measurement_validity_and_target_are_separate(self):
        m = self.load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'results.csv').write_text('trial_id,phase,status,discharge_deg,discharge_measurement_valid,repose_deg,repose_measurement_valid,rolling_deg,rolling_measurement_valid,accepted\n1,doe,complete,30,True,40,True,nan,False,False\n2,refinement,complete,31,True,41,True,42,True,True\n')
            data = m.snapshot(root)
            self.assertEqual(data['measurements']['doe']['discharge'], 1)
            self.assertEqual(data['measurements']['doe']['rolling'], 0)
            self.assertEqual(data['measurements']['refinement']['rolling'], 1)
            self.assertEqual(data['accepted'], 1)

    def test_adaptive_journal_records_terminal_and_pending(self):
        m = self.load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'adaptive_proposals.jsonl').write_text('{"event":"reserved","trial_id":100}\n{"event":"reserved","trial_id":101}\n{"event":"completed","trial_id":100}\n')
            data = m.snapshot(root)
            self.assertEqual(data['counts']['complete'], 1)
            self.assertEqual([r['trial_id'] for r in data['active']], [101])
            self.assertEqual(data['active'][0]['phase'], 'refinement')

if __name__ == '__main__':
    unittest.main()
