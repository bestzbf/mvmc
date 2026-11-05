import hashlib
import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('fullprep',Path(__file__).with_name('prepare_missing.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_case_content_unchanged_and_input_copy_inside_manifest(tmp_path):
    p=tmp_path/'source';p.mkdir();source=p/'input.npz';source.write_bytes(b'frozen input bytes')
    case={'id':'panel/case','input':'input.npz','input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'frames':150,'queries':384}
    original={'cases':[case],'protocol':{'release_binding':{'coverage_verified':True}}}
    mf=p/'manifest.json';mf.write_text('{}')
    result=module.derive_manifest(original,mf,['panel/case'],tmp_path/'new/inputs')
    new=result['cases'][0]
    assert {k:v for k,v in new.items() if k!='input'}=={k:v for k,v in case.items() if k!='input'}
    assert (tmp_path/'new'/new['input']).read_bytes()==source.read_bytes()
    assert not (tmp_path/'new'/new['input']).is_symlink()
    assert original['cases'][0]['input']=='input.npz'
    assert original['protocol']['release_binding']['coverage_verified'] is True
    assert result['protocol']['release_binding']['coverage_verified'] is False


def test_noncanonical_or_duplicate_ids_refused(tmp_path):
    for ids in [['missing'],['a','a']]:
        with pytest.raises(ValueError,match='exact frozen'):
            module.derive_manifest({'cases':[{'id':'a'}]},tmp_path/'mf.json',ids,tmp_path/'new/inputs')


def test_shared_sha_cache_reused_without_overwriting_source(tmp_path):
    source=tmp_path/'source.npz';source.write_bytes(b'original frozen bytes');before=source.stat().st_mtime_ns
    expected=hashlib.sha256(source.read_bytes()).hexdigest()
    original={'cases':[{'id':'a','input':str(source),'input_sha256':expected,'frames':64,'queries':256}]}
    manifest=tmp_path/'manifest.json';manifest.write_text('{}');cache=tmp_path/'cache'
    first=module.derive_manifest(original,manifest,['a'],tmp_path/'one/inputs',cache)
    second=module.derive_manifest(original,manifest,['a'],tmp_path/'two/inputs',cache)
    assert len(list(cache.rglob('*.npz')))==1 and not list(cache.rglob('*.tmp.*'))
    assert (tmp_path/'one'/first['cases'][0]['input']).stat().st_ino==(tmp_path/'two'/second['cases'][0]['input']).stat().st_ino
    assert source.read_bytes()==b'original frozen bytes' and source.stat().st_mtime_ns==before
