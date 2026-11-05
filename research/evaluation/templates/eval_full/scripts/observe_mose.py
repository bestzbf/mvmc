#!/usr/bin/env python3
"""Run an unchanged MoSE adapter with external JSONL phase observations.

No tensor contents are inspected, no CUDA sync is inserted, and no producer
artifact is changed. Timings are Python host wall times, not CUDA event times.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time


def run(adapter, logfile, argv, native_model=None, baseline_root=None):
    adapter = Path(adapter).resolve(strict=True)
    logfile = Path(logfile).resolve()
    if logfile.exists():
        raise FileExistsError(logfile)
    native_model = Path(native_model).resolve(strict=True) if native_model else None
    baseline_root = Path(baseline_root).resolve(strict=True) if baseline_root else None
    started = {}
    source_count = 0
    old_argv, old_profile = sys.argv, sys.getprofile()
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('x') as stream:
        def emit(event, **fields):
            stream.write(json.dumps({'at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'monotonic_seconds': time.monotonic(), 'pid': os.getpid(),
                'event': event, **fields}, ensure_ascii=False) + '\n')
            stream.flush()

        def profile(frame, event, arg):
            nonlocal source_count
            if event not in ('call', 'return'):
                return
            code = frame.f_code
            name, filename = code.co_name, Path(code.co_filename)
            fields = {}
            phase = None
            if filename == adapter and name in ('checkpoint_record', 'predict_case', 'callback'):
                phase = name
                if name == 'callback':
                    fields = {'case_id': frame.f_locals.get('case', {}).get('id'),
                              'case_index': frame.f_locals.get('index')}
            elif native_model and filename == native_model and name == 'forward':
                phase = 'native_full_sequence_forward'
                fields['query_frame_idx'] = frame.f_locals.get('query_frame_idx')
                if event == 'call':
                    source_count += 1
                    fields['forward_ordinal'] = source_count
                imgs = frame.f_locals.get('imgs')
                if event == 'call' and imgs is not None:
                    fields['input_shape'] = list(imgs.shape)
            elif name == 'from_pretrained' and getattr(frame.f_locals.get('cls'), '__name__', '') == 'MoSE3':
                phase = 'from_pretrained'
            elif baseline_root and filename.is_relative_to(baseline_root) and name in (
                    'run_adapter_predictions', '_prediction_specs', 'read_inputs',
                    '_atomic_prediction_file', '_atomic_prediction_json'):
                phase = name
            if phase is None:
                return
            ident = id(frame)
            if event == 'call':
                started[ident] = time.monotonic()
                emit(phase + '_start', **fields)
            else:
                begin = started.pop(ident, None)
                emit(phase + '_return', host_wall_seconds_no_cuda_sync=(time.monotonic()-begin if begin is not None else None), **fields)

        emit('adapter_start', adapter=str(adapter), adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),
             argv=list(argv), wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
             timing_contract='Host call/return observations; return also occurs on exception unwinding and is not a success receipt. No CUDA synchronization added.')
        try:
            sys.argv = [str(adapter), *argv]
            sys.setprofile(profile)
            result = runpy.run_path(str(adapter), run_name='__main__')
        except BaseException as exc:
            emit('adapter_exception', exception_type=type(exc).__name__, message=str(exc))
            raise
        else:
            emit('adapter_return')
            return result
        finally:
            sys.setprofile(old_profile)
            sys.argv = old_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', required=True, type=Path)
    parser.add_argument('--log', required=True, type=Path)
    parser.add_argument('--native-model', type=Path)
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('adapter_argv', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.adapter_argv[1:] if args.adapter_argv[:1] == ['--'] else args.adapter_argv
    run(args.adapter, args.log, argv, args.native_model, args.baseline_root)


if __name__ == '__main__':
    main()
