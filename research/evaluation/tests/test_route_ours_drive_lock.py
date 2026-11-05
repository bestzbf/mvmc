import importlib.util
import json
import os
from pathlib import Path
import types

import pytest

FILE=Path(__file__).with_name('route_ours_drive_lock.py')
spec=importlib.util.spec_from_file_location('ours_route',FILE)
route=importlib.util.module_from_spec(spec);spec.loader.exec_module(route)
UUID3='GPU-00000000-0000-0000-0000-000000000003'
UUID4='GPU-00000000-0000-0000-0000-000000000004'


@pytest.fixture
def parent_lock(tmp_path,monkeypatch):
    path=tmp_path/'gpu3.lock';path.write_text('')
    descriptor=os.open(path,os.O_RDWR)
    monkeypatch.setattr(route,'ROUTES',{UUID3:(3,path),UUID4:(4,tmp_path/'gpu4.lock')})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',UUID3);monkeypatch.setenv('BASELINE_GPU_LOCK_FD',str(descriptor))
    info=os.fstat(descriptor)
    text=f'pos:\t0\nlock:\t1: FLOCK ADVISORY WRITE {os.getppid()} {os.major(info.st_dev):x}:{os.minor(info.st_dev):x}:{info.st_ino} 0 EOF\n'
    original=Path.read_text
    def read_text(path,*args,**kwargs):
        if str(path)==f'/proc/self/fdinfo/{descriptor}':return text
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read_text)
    yield descriptor,path,info,text
    os.close(descriptor)


def test_exact_uuid_fd_inode_and_parent_flock(parent_lock):
    result=route.validate_parent_lock()
    assert result['physical_gpu']==3 and result['inherited_fd']==parent_lock[0]
    assert result['kernel_lock']['owner_pid']==os.getppid()


@pytest.mark.parametrize('uuid',['3','0','GPU-00000000-0000-0000-0000-000000000001',UUID3+','+UUID4])
def test_unapproved_or_ambiguous_gpu_rejected(parent_lock,monkeypatch,uuid):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',uuid)
    with pytest.raises(RuntimeError,match='exact allowed'):route.validate_parent_lock()


def test_wrong_lock_inode_rejected(parent_lock,monkeypatch,tmp_path):
    other=tmp_path/'other.lock';other.write_text('')
    monkeypatch.setattr(route,'ROUTES',{UUID3:(3,other)})
    with pytest.raises(RuntimeError,match='inode'):route.validate_parent_lock()


@pytest.mark.parametrize('raw',['','-1','0','abc'])
def test_bad_or_missing_fd_rejected(parent_lock,monkeypatch,raw):
    monkeypatch.setenv('BASELINE_GPU_LOCK_FD',raw)
    with pytest.raises(RuntimeError,match='valid inherited'):route.validate_parent_lock()


@pytest.mark.parametrize('change',[lambda x:x.replace('WRITE','READ'),lambda x:x.replace(' 0 EOF',' 1 EOF'),lambda x:'',lambda x:x.replace(str(os.getppid()),'99999999',1)])
def test_unowned_or_nonexclusive_kernel_lock_rejected(parent_lock,change):
    with pytest.raises(RuntimeError,match='parent-held'):
        route.flock_evidence(change(parent_lock[3]),parent_lock[2],os.getppid())


def test_only_one_status_constant_changes(tmp_path,monkeypatch):
    old=str(tmp_path/'old.json');monkeypatch.setattr(route,'LEGACY_STATUS',old)
    code=compile('def main():\n value='+repr(old)+'\n return (value, 8, 134400)\n','frozen','exec')
    namespace={};exec(code,namespace);original=namespace['main'].__code__
    changed,proof=route.redirect_one_status_constant(original,tmp_path/'new.json')
    assert changed.co_code==original.co_code
    assert types.FunctionType(changed,{})()==(str(tmp_path/'new.json'),8,134400)
    assert proof['co_code_identical'] and proof['all_other_code_attributes_and_constants_identical']
    assert namespace['main']()==(old,8,134400)


def test_missing_status_literal_refused(tmp_path):
    def main():return 134400
    with pytest.raises(RuntimeError,match='exactly one'):route.redirect_one_status_constant(main.__code__,tmp_path/'new')


def test_execute_preserves_argv_old_status_source_and_restores_code(parent_lock,tmp_path,monkeypatch):
    old=tmp_path/'old_status.json';old.write_text('old evidence')
    monkeypatch.setattr(route,'LEGACY_STATUS',str(old))
    source=tmp_path/'frozen.py'
    source.write_text('from pathlib import Path\nimport sys\nfrom contextlib import contextmanager\n'
        '@contextmanager\ndef inference_lock(path):\n raise AssertionError("original lock must be routed")\n yield\n'
        'def main():\n with inference_lock(Path('+repr(str(route.LEGACY_LOCK))+')):\n'
        '  Path('+repr(str(old))+').write_text("new job only")\n  return list(sys.argv)\n')
    monkeypatch.setattr(route,'SOURCE',source);monkeypatch.setattr(route,'SOURCE_SHA',route.sha(source))
    before=source.read_bytes();source_sha=route.sha(source);saved_argv=list(route.sys.argv)
    original_run=route.runpy.run_path;seen={}
    def load(*args,**kwargs):
        namespace=original_run(*args,**kwargs);seen.update(main=namespace['main'],code=namespace['main'].__code__,guard=namespace['inference_lock']);return namespace
    monkeypatch.setattr(route.runpy,'run_path',load)
    receipt=tmp_path/'new_job/routing.json';argv=['predict','--manifest','/new/manifest.json','--output','/new/predictions']
    result=route.execute(argv,receipt)
    assert result==[str(source),*argv] and route.sys.argv==saved_argv
    assert source.read_bytes()==before and route.sha(source)==source_sha
    assert old.read_text()=='old evidence'
    assert receipt.with_name(receipt.stem+'.native_streamed_status.json').read_text()=='new job only'
    assert seen['main'].__code__ is seen['code'] and seen['main'].__globals__['inference_lock'] is seen['guard']
    value=json.loads(receipt.read_text());assert value['phase']=='native_main_returned'
    assert value['status_IO_redirection']['co_code_identical']
    first=receipt.read_bytes()
    assert route.execute(argv,receipt)==[str(source),*argv]
    assert receipt.read_bytes()==first
    attempts=list(receipt.parent.glob('routing.attempt_*.json'))
    attempts=[p for p in attempts if '.native_streamed_status.' not in p.name]
    assert len(attempts)==1 and json.loads(attempts[0].read_text())['previous_anchor_receipt']['sha256']==route.sha(receipt)
    assert old.read_text()=='old evidence'


def test_status_artifact_must_not_be_overwritten(parent_lock,tmp_path,monkeypatch):
    source=tmp_path/'source.py';source.write_text('')
    monkeypatch.setattr(route,'SOURCE',source);monkeypatch.setattr(route,'SOURCE_SHA',route.sha(source))
    receipt=tmp_path/'routing.json';receipt.with_name('routing.native_streamed_status.json').write_text('prior')
    with pytest.raises(FileExistsError,match='prior process'):route.execute(['predict'],receipt)
    assert receipt.with_name('routing.native_streamed_status.json').read_text()=='prior'


def test_frozen_source_sha_change_refused(parent_lock,tmp_path,monkeypatch):
    path=tmp_path/'bad.py';path.write_text('bad')
    monkeypatch.setattr(route,'SOURCE',path)
    with pytest.raises(RuntimeError,match='source SHA'):route.execute(['predict'],tmp_path/'routing.json')
