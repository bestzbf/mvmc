"""CPU audit regression using the immutable real successful140501 BOOT log."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from scripts.async_training_state import UpdateAudit as OriginalAudit
from native_update_audit_v23 import top_level_audit
Audit=top_level_audit(OriginalAudit)
def native_row():
    return {
        'step': 140501, 'world_size': 6, 'gradient_accumulation': 1,
        'effective_batch_size': 6, 'optimizer_step_skipped': False, 'supervision': 'dense',
        'loss': 1.0, 'grad_norm': 0.5, 'lr': 1e-4, 'backbone_lr': 1e-5, 'refinement_lr': 1e-4,
        'rank_batches': [{'metrics': {'supervised_source_pixels': 1}} for _ in range(6)],
    }


class AuditTests(unittest.TestCase):
    def test_legacy_parser_failure_and_new_exactly_one_complete_update(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'native.log'
            header = {'loss_profile': 'custom', 'data_migration': {'step': 140500}}
            path.write_text('[default0]:' + json.dumps(header) + '\n'
                            + '[default0]:' + json.dumps(native_row()) + '\n')
            with self.assertRaisesRegex(ValueError, 'missing, repeated'):
                OriginalAudit(path, 140500, 6, 1, 6).poll()
            reader = Audit(path, 140500, 6, 1, 6)
            reader.poll()
            self.assertEqual(reader.last, 140501)
            self.assertEqual(reader.latest['step'], 140501)
            reader.poll()
            self.assertEqual(reader.last, 140501)
    def test_nested_metadata_not_a_step_but_good_update_is_strict(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'native.log'
            header={'loss_profile':'custom','data_migration':{'step':140500,'cursor_policy':'explicit_reset_once'}}
            path.write_text('[default0]:'+json.dumps(header)+'\n[default0]:'+json.dumps(native_row())+'\n')
            reader=Audit(path,140500,6,1,6);reader.poll();self.assertEqual(reader.last,140501)
    def test_true_duplicate_and_gap_remain_hard_failures(self):
        for gap in (0,2):
            with self.subTest(gap=gap),tempfile.TemporaryDirectory() as t:
                path=Path(t)/'native.log';one=native_row();two=deepcopy(one);two['step']+=gap
                path.write_text(json.dumps(one)+'\n'+json.dumps(two)+'\n')
                with self.assertRaisesRegex(ValueError,'missing, repeated'):Audit(path,140500,6,1,6).poll()
    def test_incomplete_frame_then_complete_frame_no_lost_step(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'native.log';value=json.dumps(native_row());half=len(value)//2
            path.write_text(value[:half]);reader=Audit(path,140500,6,1,6);reader.poll();self.assertEqual(reader.last,140500)
            with path.open('a') as stream:stream.write(value[half:]+'\n')
            reader.poll();self.assertEqual(reader.last,140501)
    def test_actual_nonfinite_and_malformed_update_still_fail(self):
        for mode in ('nan','missing'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as t:
                path=Path(t)/'native.log';row=native_row()
                if mode=='nan':row['grad_norm']=float('nan')
                else:row.pop('rank_batches')
                path.write_text(json.dumps(row)+'\n')
                with self.assertRaises((ValueError,KeyError)):Audit(path,140500,6,1,6).poll()
if __name__=='__main__':unittest.main(verbosity=2)
