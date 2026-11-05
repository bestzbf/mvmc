import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


PATH = Path(__file__).with_name('observe_mose.py')
spec = importlib.util.spec_from_file_location('mose_observer', PATH)
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


def test_original_argv_outputs_preserved_and_forward_observed(tmp_path):
    native = tmp_path / 'fake_native.py'
    native.write_text('class MoSE3:\n    def forward(self, imgs, query_frame_idx=0):\n        return query_frame_idx * 3\n')
    adapter = tmp_path / 'adapter.py'
    adapter.write_text('import json,sys\nfrom fake_native import MoSE3\nclass Images:\n    shape=(1,300,3,518,518)\ndef callback(inputs,case,index):\n    return sum(MoSE3().forward(Images(),query_frame_idx=q) for q in [1,3])\nif __name__=="__main__":\n    print(json.dumps({"argv":sys.argv[1:],"value":callback(None,{"id":"caseA"},0)}))\n')
    original = subprocess.run([sys.executable, str(adapter), '--resume', 'literal'], text=True, capture_output=True, check=True)
    log = tmp_path / 'events.jsonl'
    wrapped = subprocess.run([sys.executable, str(PATH), '--adapter', str(adapter), '--native-model', str(native), '--log', str(log), '--', '--resume', 'literal'], env={**__import__('os').environ, 'PYTHONPATH': str(tmp_path)}, text=True, capture_output=True, check=True)
    assert json.loads(original.stdout) == json.loads(wrapped.stdout)
    events = [json.loads(line) for line in log.read_text().splitlines()]
    starts = [e for e in events if e['event'] == 'native_full_sequence_forward_start']
    assert [e['query_frame_idx'] for e in starts] == [1, 3]
    assert all(e['input_shape'] == [1, 300, 3, 518, 518] for e in starts)
    assert events[-1]['event'] == 'adapter_return'
    assert any(e['event'] == 'callback_start' and e['case_id'] == 'caseA' for e in events)


def test_exception_and_global_profile_argv_restored(tmp_path):
    adapter = tmp_path / 'bad_adapter.py'
    adapter.write_text('raise ValueError("unchanged-model-error")\n')
    argv, profile = sys.argv, sys.getprofile()
    log = tmp_path / 'events.jsonl'
    with pytest.raises(ValueError, match='unchanged-model-error'):
        observer.run(adapter, log, ['--resume'])
    assert sys.argv is argv and sys.getprofile() is profile
    assert json.loads(log.read_text().splitlines()[-1])['event'] == 'adapter_exception'
    with pytest.raises(FileExistsError):
        observer.run(adapter, log, [])
