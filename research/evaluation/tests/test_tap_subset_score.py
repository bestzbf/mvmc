import copy,json
from pathlib import Path
import numpy as np
import pytest
import tap_subset_score as s


def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2)+'\n')

@pytest.fixture
def caches(tmp_path):
 cp={'kind':'author_rgb_only_native_baseline','method':'TestModel','sha256':'c'*64,'native_multicamera4rc_loading':False}
 cases=[{'id':f'drivetrack/case{i}.npz','input':f'inputs/{i}.npz','input_sha256':str(i+1)*64,'subset':'drivetrack','frames':1,'queries':1,'views':[0]} for i in range(3)]
 original={'schema':'released-tracking-benchmark-v1','benchmark':'tapvid3d','input_condition':'rgb_only','coordinates':'per_frame_camera','protocol':{'subsets':['drivetrack'],'split':'minival','scaling':'median'},'official_sources':{'commit':'test'},'cases':cases}
 def source(name,indices,settings=None):
  root=tmp_path/name;root.mkdir();sm=copy.deepcopy(original);sm['cases']=[copy.deepcopy(cases[i]) for i in indices]
  for c in sm['cases']:c['input']='relocated/'+Path(c['input']).name
  manifest=root/'manifest.json';write(manifest,sm);state={'schema':'released-baseline-run-v1','manifest_sha256':s.sha(manifest),'checkpoint':cp,'inference_settings':settings or {'max_side':518,'numerics':'bf16'},'runtime_versions':{'torch':'same'},'numerics':{'device':'B300'}};write(root/'run-state.json',state);records=[]
  for index,c in enumerate(sm['cases']):
   payload=root/f'{index:06d}.npz';arrays={'tracks_XYZ':np.ones((1,1,3),np.float32),'visibility':np.ones((1,1),bool)};np.savez_compressed(payload,**arrays)
   record={'id':c['id'],'file':payload.name,'sha256':s.sha(payload),'input_sha256':c['input_sha256'],'model':'TestModel'};records.append(record)
   write(payload.with_suffix('.receipt.json'),{'schema':'released-prediction-case-v1','record':record,'case_sha256':s.canonical(c),'run_state_sha256':s.canonical(state),'arrays':{k:{'shape':list(v.shape),'dtype':v.dtype.str} for k,v in arrays.items()}})
  ip=root/'predictions.json';write(ip,{'schema':'released-tracking-predictions-v1','manifest_sha256':s.sha(manifest),'checkpoint':cp,'benchmark':sm['benchmark'],'input_condition':sm['input_condition'],'coordinates':sm['coordinates'],'cases':records,'run_state_sha256':s.canonical(state)})
  return {'kind':'prediction_index','path':str(ip),'manifest':str(manifest),'manifest_sha256':s.sha(manifest),'index_sha256':s.sha(ip)}
 return cp,cases,original,source


def call(caches,sources,selected=None):
 cp,cases,original,_=caches;ids=selected or [c['id'] for c in cases[:2]]
 return s.load_sources({'expected_checkpoint':cp,'sources':sources},original,ids)


def test_disjoint_delta_manifests_merge_without_copies(caches):
 cp,cases,original,make=caches;a=make('a',[0]);b=make('b',[1]);paths=[Path(a['path']).parent/'000000.npz',Path(b['path']).parent/'000000.npz'];before=[p.read_bytes() for p in paths]
 catalog,evidence,signature=call(caches,[a,b]);assert set(catalog)=={cases[0]['id'],cases[1]['id']};assert len(evidence)==2 and all(x['settings_signature']==signature for x in evidence);assert [p.read_bytes() for p in paths]==before;assert {v['path'] for v in catalog.values()}=={str(p) for p in paths}


def test_settings_mismatch_rejected(caches):
 make=caches[3];a=make('a',[0]);b=make('b',[1],{'max_side':504,'numerics':'bf16'})
 with pytest.raises(ValueError,match='settings/runtime'):call(caches,[a,b])


def test_duplicate_case_rejected_and_explicit_source_selection_works(caches):
 make=caches[3];a=make('a',[0,1]);b=make('b',[0,1])
 with pytest.raises(ValueError,match='Ambiguous duplicate'):call(caches,[a,b])
 a['case_ids']=[caches[1][0]['id']];b['case_ids']=[caches[1][1]['id']];assert len(call(caches,[a,b])[0])==2


def test_wrong_checkpoint_rejected(caches):
 a=caches[3]('a',[0,1]);p=Path(a['path']);idx=s.read(p);idx['checkpoint']=dict(idx['checkpoint'],sha256='d'*64);write(p,idx);a.pop('index_sha256')
 with pytest.raises(ValueError,match='checkpoint'):call(caches,[a])


def test_modified_payload_rejected(caches):
 a=caches[3]('a',[0,1]);p=Path(a['path']).parent/'000000.npz';p.write_bytes(p.read_bytes()+b'changed')
 with pytest.raises(ValueError,match='bytes differ'):call(caches,[a])


def test_wrong_case_receipt_rejected(caches):
 a=caches[3]('a',[0,1]);p=Path(a['path']).parent/'000000.receipt.json';r=s.read(p);r['case_sha256']='f'*64;write(p,r)
 with pytest.raises(ValueError,match='case identity'):call(caches,[a])


def test_escape_original_prediction_root_rejected(caches,tmp_path):
 a=caches[3]('a',[0,1]);p=Path(a['path']);idx=s.read(p);outside=tmp_path/'outside.npz';outside.write_bytes((p.parent/'000000.npz').read_bytes());idx['cases'][0]['file']='../outside.npz';write(p,idx);a.pop('index_sha256')
 with pytest.raises(ValueError,match='escaped'):call(caches,[a])


def test_missing_case_never_becomes_zero(caches):
 a=caches[3]('a',[0])
 with pytest.raises(ValueError,match='coverage incomplete'):call(caches,[a])


def test_partial_committed_receipts_without_fake_complete_index(caches):
 a=caches[3]('partial',[0,1,2]);root=Path(a['path']).parent;(root/'predictions.json').unlink();(root/'000002.npz').unlink();(root/'000002.receipt.json').unlink();a.update(kind='committed_case_receipts',path=str(root));a.pop('index_sha256');assert len(call(caches,[a])[0])==2
