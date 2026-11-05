#!/usr/bin/env python3
"""Run the unchanged step134400 streamed main under a verified parent GPU lock.

Only the original main's inference_lock global and one diagnostic status-path
constant are replaced. Model/decoder bytecode, argv and producer state are unchanged.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import marshal
import os
from pathlib import Path
import runpy
import stat
import sys
import time

SOURCE=Path('@@OURS_STREAMED_ROOT@@/run_streamed_released_134400.py')
SOURCE_SHA='345d50a1e0300f54a5e55ae7a82a572ff8d980ac1c72d718fb09456c3d5844bc'
ROUTES={
    '@@GPU3_UUID@@':(3,Path('@@BENCHMARK_ROOT@@/evaluation/.gpu3_single_inference.lock')),
    '@@GPU4_UUID@@':(4,Path('@@BENCHMARK_ROOT@@/evaluation/.gpu4_single_inference.lock')),
}
LEGACY_LOCK=Path('@@BENCHMARK_ROOT@@/evaluation/.gpu4_single_inference.lock')
LEGACY_STATUS=str(SOURCE.parent/'drive_minival_134400_streamed_status.json')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def redirect_one_status_constant(code,new_status):
    constants=list(code.co_consts)
    matches=[i for i,v in enumerate(constants) if isinstance(v,str) and v==LEGACY_STATUS]
    if len(matches)!=1:raise RuntimeError('Expected exactly one frozen diagnostic status literal')
    new_status=str(Path(new_status).resolve())
    if new_status==LEGACY_STATUS:raise RuntimeError('New status must preserve the old diagnostic artifact')
    constants[matches[0]]=new_status
    changed=code.replace(co_consts=tuple(constants))
    for attribute in dir(code):
        if attribute.startswith('co_') and attribute!='co_consts' and not callable(getattr(code,attribute)):
            if getattr(code,attribute)!=getattr(changed,attribute):raise AssertionError('Non-constant code metadata changed: '+attribute)
    if changed.co_code!=code.co_code:raise AssertionError('Model/decoder bytecode changed')
    for i,(old,new) in enumerate(zip(code.co_consts,changed.co_consts)):
        if i!=matches[0] and old is not new:raise AssertionError('Another code constant changed')
    return changed,{'constant_index':matches[0],'old_status':LEGACY_STATUS,'new_status':new_status,
        'original_main_code_sha256':hashlib.sha256(marshal.dumps(code)).hexdigest(),
        'routed_main_code_sha256':hashlib.sha256(marshal.dumps(changed)).hexdigest(),
        'co_code_identical':True,'all_other_code_attributes_and_constants_identical':True}


def flock_evidence(text, descriptor_stat, parent_pid):
    """Kernel fdinfo lists only locks attached to this open file description."""
    for line in text.splitlines():
        parts=line.split()
        if len(parts)!=9 or parts[0]!='lock:':continue
        if parts[2:5]!=['FLOCK','ADVISORY','WRITE'] or parts[7:]!=['0','EOF']:continue
        major,minor,inode=parts[6].split(':')
        if (int(major,16),int(minor,16),int(inode))!=(os.major(descriptor_stat.st_dev),os.minor(descriptor_stat.st_dev),descriptor_stat.st_ino):continue
        if int(parts[5])!=parent_pid:continue
        return {'type':'FLOCK','mode':'WRITE','whole_file':True,'owner_pid':int(parts[5]),
                'inode':descriptor_stat.st_ino,'device':descriptor_stat.st_dev}
    raise RuntimeError('Inherited descriptor lacks a parent-held exclusive whole-file FLOCK in kernel fdinfo')


def validate_parent_lock():
    uuid=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if uuid not in ROUTES:raise RuntimeError('CUDA_VISIBLE_DEVICES must be the exact allowed GPU3/4 UUID')
    physical,path=ROUTES[uuid]
    raw=os.environ.get('BASELINE_GPU_LOCK_FD','')
    if not raw.isdecimal() or int(raw)<3:raise RuntimeError('Missing valid inherited BASELINE_GPU_LOCK_FD')
    descriptor=int(raw)
    info=os.fstat(descriptor);expected=path.stat()
    if not stat.S_ISREG(info.st_mode) or not stat.S_ISREG(expected.st_mode):raise RuntimeError('GPU lock descriptor/path is not a regular file')
    if (info.st_dev,info.st_ino)!=(expected.st_dev,expected.st_ino):raise RuntimeError('Inherited FD is not the selected physical GPU lock inode')
    proof=flock_evidence(Path(f'/proc/self/fdinfo/{descriptor}').read_text(),info,os.getppid())
    return {'physical_gpu':physical,'gpu_uuid':uuid,'lock_path':str(path),'inherited_fd':descriptor,
            'kernel_lock':proof,'parent_pid':os.getppid()}


def update_receipt(path,receipt):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with temp.open('x') as stream:
        json.dump(receipt,stream,ensure_ascii=False,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def execute(native_args,receipt_path):
    if not native_args or native_args[0]!='predict':raise ValueError('This routing wrapper is only for frozen streamed prediction')
    if sha(SOURCE)!=SOURCE_SHA:raise RuntimeError('Frozen streamed wrapper source SHA changed')
    requested_receipt=receipt_path
    previous=None
    if receipt_path.exists() or receipt_path.is_symlink():
        if receipt_path.is_symlink():raise RuntimeError('Routing receipt anchor cannot be a symlink')
        previous={'path':str(receipt_path),'sha256':sha(receipt_path)}
        receipt_path=receipt_path.with_name(receipt_path.stem+f'.attempt_{os.getpid()}_{time.time_ns()}'+receipt_path.suffix)
    new_status=receipt_path.with_name(receipt_path.stem+'.native_streamed_status.json')
    if receipt_path.exists() or receipt_path.is_symlink() or new_status.exists() or new_status.is_symlink():
        raise FileExistsError('Preserve prior process routing/status artifacts')
    initial=validate_parent_lock()
    receipt={'schema':'ours134400-drive-parent-lock-routing-v1','pid':os.getpid(),'started_epoch':time.time(),
             'source_wrapper':str(SOURCE),'source_wrapper_sha256':SOURCE_SHA,'routing_wrapper_sha256':sha(__file__),
             'native_argv':[str(SOURCE),*native_args],'initial_route':initial,'phase':'routing_validated',
             'requested_receipt_anchor':str(requested_receipt),'actual_attempt_receipt':str(receipt_path),
             'previous_anchor_receipt':previous,
             'changes':'Only original main global inference_lock and one diagnostic status IO literal; no fstat/inode/source/model/producer-state override',
             'old_diagnostic_status_preserved':LEGACY_STATUS}
    update_receipt(receipt_path,receipt)
    before_argv=list(sys.argv);before_path=list(sys.path)
    sys.argv=[str(SOURCE),*native_args];sys.path.insert(0,str(SOURCE.parent))
    original_main=None;original_guard=None;original_code=None
    try:
        namespace=runpy.run_path(str(SOURCE),run_name='_frozen_ours134400_streamed_routed')
        original_main=namespace['main'];globals_=original_main.__globals__
        original_guard=globals_['inference_lock']
        original_code=original_main.__code__
        original_main.__code__,proof=redirect_one_status_constant(original_code,new_status)
        receipt['status_IO_redirection']=proof
        @contextmanager
        def routed_lock(requested_path):
            if Path(requested_path)!=LEGACY_LOCK:raise RuntimeError('Frozen main requested an unexpected original lock path')
            actual=validate_parent_lock()
            if actual!=initial:raise RuntimeError('Parent lock route changed before native inference')
            receipt['phase']='native_lock_entered';receipt['lock_entry_epoch']=time.time();update_receipt(receipt_path,receipt)
            try:yield actual['inherited_fd']
            finally:
                if validate_parent_lock()!=initial:raise RuntimeError('Parent lock route changed during native inference')
        globals_['inference_lock']=routed_lock
        receipt['patch_scope']='one original globals entry inference_lock + one exact diagnostic status IO constant'
        result=original_main()
        receipt.update(phase='native_main_returned',finished_epoch=time.time());update_receipt(receipt_path,receipt)
        return result
    except BaseException as error:
        receipt.update(phase='failed',finished_epoch=time.time(),error=type(error).__name__+': '+str(error));update_receipt(receipt_path,receipt)
        raise
    finally:
        if original_main is not None and original_guard is not None:original_main.__globals__['inference_lock']=original_guard
        if original_main is not None and original_code is not None:original_main.__code__=original_code
        sys.argv=before_argv;sys.path[:]=before_path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--routing-receipt',type=Path,required=True)
    parser.add_argument('native_args',nargs=argparse.REMAINDER);args=parser.parse_args()
    argv=args.native_args[1:] if args.native_args[:1]==['--'] else args.native_args
    execute(argv,args.routing_receipt)


if __name__=='__main__':main()
