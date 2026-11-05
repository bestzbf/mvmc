"""CPU synthetic filesystem/serialization tests, NOT valid CUDA RNG generation.

Successful filesystem tests mock only the mandatory external CUDA receipt check;
all source/staged/output serialization and value comparisons remain real CPU.
"""
from copy import deepcopy
import ast
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
import torch

import publish_expanded_checkpoint_v21 as pub
import test_expand_checkpoint_v14 as fixtures

TARGET = 'c'*64


class PublisherTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        # NFS cleanup happens in the outer harness after this process exits:
        # exception tracebacks can retain read-only mmap storages until then.
        self.root=Path(tempfile.mkdtemp(prefix='publisher_test_'))
        self.source=fixtures.payload();self.source['step']=140300
        self.source['training_state'].update(optimizer_updates=140300,batches_consumed=25)
        self.source_path=self.root/'source.pt';torch.save(self.source,self.source_path)
        self.source_sha=pub.sha(self.source_path)
        self.envelope=fixtures.envelope()
        self.envelope['provenance']['source_checkpoint_sha256']=self.source_sha
        self.rng_path=self.root/'rng.pt';torch.save(self.envelope,self.rng_path)
        self.rng_sha=pub.sha(self.rng_path)
        self.proof_path=self.root/'proof.json';self.proof_path.write_text('{}')
        self.proof_sha=pub.sha(self.proof_path)
    def tearDown(self):pass
    def verifier(self,stage):
        return pub.verify_file(self.source_path,self.source_sha,self.rng_path,self.rng_sha,
            stage,'new_data_stream_reset',TARGET)
    def publish(self,name='bundle'):
        with patch.object(pub,'check_rng_proof',return_value=None):
            return pub.publish(self.source_path,self.source_sha,self.rng_path,self.rng_sha,
                self.proof_path,self.proof_sha,self.root/name,'new_data_stream_reset',TARGET,
                _test_independent_verifier=self.verifier)
    def test_newstream_unaligned_archives_all_old_state_without_changing_values(self):
        before=deepcopy(self.source)
        with self.assertRaises(ValueError):pub.plan_for(self.source,'zero_skip_old_stream')
        result=pub.transformed(self.source,self.envelope,self.source_sha,'new_data_stream_reset',TARGET)
        pub.expand._equal(before,self.source,'source_immutable')
        pub.expand._equal(before['training_state'],result[pub.expand.LINEAGE]['old_training_state'])
        for key in ('model','optimizer','config','step','grad_scaler'):
            pub.expand._equal(before[key],result[key],key)
        self.assertEqual(result['training_state']['optimizer_updates'],140300)
        self.assertEqual((result['training_state']['data_epoch'],result['training_state']['batches_consumed']),(0,0))
        self.assertFalse(result[pub.expand.LINEAGE]['plan']['old_stream_continuity_claimed'])
        self.assertEqual(result['training_state']['data_config_sha256'],before['training_state']['data_config_sha256'])
    def test_newstream_requires_archived_original_data_fingerprint(self):
        value=deepcopy(self.source)
        value['training_state'].pop('data_config_sha256')
        with self.assertRaisesRegex(ValueError,'external admitted original data SHA256'):
            pub.plan_for(value,'new_data_stream_reset',TARGET)
        config=self.root/'old_config.json';config.write_text('{"frozen_old_config":true}')
        external=pub.source_data_contract(config,pub.sha(config))
        result=pub.transformed(value,self.envelope,self.source_sha,'new_data_stream_reset',TARGET,external)
        self.assertNotIn('data_config_sha256',result['training_state'])
        self.assertIn('missing_fingerprint',result[pub.expand.LINEAGE]['plan']['source_data_sha256_origin'])
        pub.verify_payload(value,result,self.envelope,self.source_sha,'new_data_stream_reset',TARGET,external)
    def test_actual_native_migration_then_balance_and_cursor_reset_order(self):
        tree=ast.parse((fixtures.NATIVE/'multicamera4rc/train.py').read_text())
        train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_train')
        block=next(n for n in train.body if isinstance(n,ast.If) and ast.unparse(n.test)=='args.migrate_data_config')
        from types import SimpleNamespace
        args=SimpleNamespace(migrate_data_config=True,reset_data_cursor=False)
        ns=dict(args=args,saved_fingerprint='b'*64,data_fingerprint=TARGET,resume=self.source,
                data_migration=None)
        exec(compile(ast.Module(body=[block],type_ignores=[]),'actual_native_migration','exec'),ns)
        self.assertTrue(args.reset_data_cursor)
        self.assertEqual(ns['data_migration']['step'],140300)
        self.assertEqual(ns['data_migration']['cursor_policy'],'explicit_reset_once')
        balance_block=next(n for n in train.body if isinstance(n,ast.If)
            and ast.unparse(n.test)=='balance and args.migrate_data_config')
        ns.update(balance=deepcopy(self.source['training_state']['load_balancing']),
            context=SimpleNamespace(world_size=6),dataset=range(1000))
        exec(compile(ast.Module(body=[balance_block],type_ignores=[]),'actual_native_balance','exec'),ns)
        fixtures.balance_module.validate_balance(ns['balance'],6,1,1000)
        self.assertEqual((ns['balance']['world_size'],ns['balance']['start_epoch'],ns['balance']['start_batch']),(6,0,0))
        cursor=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_resume_data_cursor')
        scope={};exec(compile(ast.Module(body=[cursor],type_ignores=[]),'actual_native_cursor','exec'),scope)
        self.assertEqual(scope['_resume_data_cursor'](self.source['training_state'],140300,1000,6,1,
            reset=args.reset_data_cursor,sampler_kind='distributed_v1'),(0,0))
        args=SimpleNamespace(migrate_data_config=False,reset_data_cursor=False)
        ns.update(args=args,saved_fingerprint=TARGET,data_fingerprint=TARGET)
        exec(compile(ast.Module(body=[block],type_ignores=[]),'actual_ordinary_migration','exec'),ns)
        self.assertEqual(ns['data_migration']['step'],140300)
        self.assertFalse(args.reset_data_cursor)
    def test_publish_full_fresh_values_commit_and_nooverwrite(self):
        receipt=self.publish();self.assertTrue(receipt['committed'])
        with patch.object(pub,'check_rng_proof',return_value=None):
            verification=pub.verify_bundle(self.root/'bundle')
        self.assertTrue(verification['exact_all_values_verified'])
        self.assertEqual(pub.sha(self.source_path),self.source_sha)
        final_sha=pub.sha(self.root/'bundle/checkpoint.pt')
        with self.assertRaises(FileExistsError):self.publish()
        self.assertEqual(pub.sha(self.root/'bundle/checkpoint.pt'),final_sha)
        with self.assertRaisesRegex(ValueError,'Commit marker differs'):
            pub.verify_bundle(self.root/'bundle','0'*64)
    def test_actual_independent_CPU_process_fresh_loads_all_values(self):
        result=pub.transformed(self.source,self.envelope,self.source_sha,'new_data_stream_reset',TARGET)
        candidate=self.root/'candidate.pt';torch.save(result,candidate)
        command=[sys.executable,'-B',str(Path(pub.__file__)), 'verify-candidate',
            '--source',str(self.source_path),'--source-sha256',self.source_sha,
            '--new-rank-rng',str(self.rng_path),'--new-rank-rng-sha256',self.rng_sha,
            '--candidate',str(candidate),'--mode','new_data_stream_reset','--target-data-sha256',TARGET]
        completed=subprocess.run(command,capture_output=True,text=True,check=True,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
        report=json.loads(completed.stdout)
        self.assertTrue(report['exact_all_values_verified']);self.assertFalse(report['cuda_initialized'])
    def test_independent_legacy_checkpoint_and_external_config_binding(self):
        value=deepcopy(self.source);value['training_state'].pop('data_config_sha256')
        path=self.root/'legacy.pt';torch.save(value,path);source_sha=pub.sha(path)
        envelope=deepcopy(self.envelope);envelope['provenance']['source_checkpoint_sha256']=source_sha
        rng=self.root/'legacy_rng.pt';torch.save(envelope,rng)
        config=self.root/'admitted_config.json';config.write_text('{"original_stream":true}')
        external=pub.source_data_contract(config,pub.sha(config))
        result=pub.transformed(value,envelope,source_sha,'new_data_stream_reset',TARGET,external)
        candidate=self.root/'legacy_candidate.pt';torch.save(result,candidate)
        command=[sys.executable,'-B',str(Path(pub.__file__)), 'verify-candidate',
            '--source',str(path),'--source-sha256',source_sha,
            '--new-rank-rng',str(rng),'--new-rank-rng-sha256',pub.sha(rng),
            '--candidate',str(candidate),'--mode','new_data_stream_reset','--target-data-sha256',TARGET,
            '--source-data-config',str(config),'--source-data-sha256',external['sha256']]
        completed=subprocess.run(command,capture_output=True,text=True,check=True,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
        report=json.loads(completed.stdout)
        self.assertTrue(report['exact_all_values_verified'])
        self.assertIn('missing_fingerprint',report['plan']['source_data_sha256_origin'])
        config.write_text('{"changed":true}')
        with self.assertRaises(ValueError):pub.require_source_data(external)
    def test_serialization_failure_has_no_final_or_commit(self):
        with patch.object(torch,'save',side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError,'disk full'):self.publish()
        self.assertFalse((self.root/'bundle/checkpoint.pt').exists())
        self.assertFalse((self.root/'bundle/COMMITTED.json').exists())
        self.assertEqual(pub.sha(self.source_path),self.source_sha)
    def test_corrupted_moment_fails_independent_full_value_verifier(self):
        value=pub.transformed(self.source,self.envelope,self.source_sha,'new_data_stream_reset',TARGET)
        value=deepcopy(value);value['optimizer']['state'][0]['exp_avg'][0,0]+=1
        candidate=self.root/'corrupt.pt';torch.save(value,candidate)
        with self.assertRaisesRegex(ValueError,'tensor changed'):
            self.verifier(candidate)
    def test_ambiguous_nfs_success_reconciled_but_wrong_existing_inode_rejected(self):
        target=self.root/'linked';real=os.link
        def ambiguous(source,destination,**kwargs):
            real(source,destination,**kwargs)
            raise OSError('NFS reply lost after success')
        with patch.object(os,'link',side_effect=ambiguous):
            self.assertTrue(pub.link_exact(self.source_path,target,self.source_sha))
        conflict=self.root/'conflict';conflict.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):pub.link_exact(self.source_path,conflict,self.source_sha)
        self.assertEqual(conflict.read_bytes(),b'keep')
    def test_marker_failure_never_commits_visible_checkpoint(self):
        original=pub.link_exact
        def fail_marker(source,destination,digest):
            if Path(destination).name=='COMMITTED.json':raise OSError('marker link failed')
            return original(source,destination,digest)
        with patch.object(pub,'link_exact',side_effect=fail_marker):
            with self.assertRaisesRegex(OSError,'marker link failed'):self.publish()
        self.assertTrue((self.root/'bundle/checkpoint.pt').exists())
        self.assertFalse((self.root/'bundle/COMMITTED.json').exists())
        with self.assertRaises(FileNotFoundError):pub.verify_bundle(self.root/'bundle')
    def test_unsupported_directory_fsync_and_synthetic_rng_proof_fail_closed(self):
        with patch.object(pub,'fsync_directory',side_effect=OSError('unsupported fsync')):
            with self.assertRaisesRegex(OSError,'unsupported fsync'):self.publish()
        self.assertFalse((self.root/'bundle/COMMITTED.json').exists())
        with self.assertRaises(ValueError):pub.check_rng_proof({},self.envelope,self.rng_sha)
    def test_zero_skip_branch_reuses_frozen_verifier(self):
        self.source['training_state']['batches_consumed']=24
        result=pub.transformed(self.source,self.envelope,self.source_sha,'zero_skip_old_stream')
        self.assertEqual(result['training_state']['batches_consumed'],20)
        pub.expand.verify_transformed_payload(self.source,result,self.envelope,source_sha256=self.source_sha)


if __name__=='__main__':unittest.main(verbosity=2)
