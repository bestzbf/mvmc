"""Metadata-only reporter eligibility/atomic-row regressions; no metric scoring."""
import importlib.util
import json
from pathlib import Path

import pytest

FILE=Path(__file__).with_name('report_full.py')
spec=importlib.util.spec_from_file_location('full_reporter',FILE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
EXCLUDED='adt_mini/fixture_ineligible.npz'


@pytest.fixture(autouse=True)
def verify_report_publication(monkeypatch):
    def check_written_reports(base):
        full = base / 'eval_full_20261105'
        assert (full/'full_results.json').is_file()
        assert (full/'全量评估对比.md').is_file()
    monkeypatch.setattr(module, 'publish_split_tables', check_written_reports)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def fixture(root,panel='trajectory_adt_mini'):
    root.mkdir()
    prefix=panel.removeprefix('trajectory_')
    ids=[f'{prefix}/fixture_{i}.npz' for i in range(50)]
    if panel=='trajectory_adt_mini':ids[39]=EXCLUDED
    eligible=[i for i in ids if i!=EXCLUDED]
    inventory={'canonical_panels':{panel:{'benchmark':'worldtrack','cases':[{'id':i} for i in ids]}},
               'rows':[],'total_missing_GPU_cases':0}
    targets=[]
    for method in ['Ours134400','UniQuery4R']:
        rid=method+'__'+panel
        inventory['rows'].append({'id':rid,'method':method,'panel_id':panel,'existing_ids':ids,'total':50})
        scorepath=root/'scores'/(rid+'.json')
        targets.append({'row_id':rid,'score_path':str(scorepath),'reused_full_score':method=='Ours134400'})
        metrics={'avg_pts_global':.8,'avg_pts_sim3_closed':.9,'epe_global':.2,'epe_sim3_closed':.1}
        if method=='UniQuery4R':metrics['avg_pts_global']=.81
        save(scorepath,{'sample_ids':ids,'scored_sample_ids':eligible,'full_frozen_coverage_verified':True,'metrics':metrics})
    save(root/'planning/full_inventory.json',inventory)
    save(root/'full_targets.json',{'rows':targets})
    save(root/'progress.json',{'records':[],'updated_at_utc':'fixed CPU fixture','overall_status':'running','final':False,'counts':{}})
    return root/'scores'/('UniQuery4R__'+panel+'.json')


def new_row(summary):return next(r for r in summary['rows'] if r['method']=='UniQuery4R')


def test_correct_world_adt_49_scored_from_50_predictions(tmp_path):
    root=tmp_path/'eval_full_20261105';fixture(root);summary=module.publish(root);row=new_row(summary)
    assert not summary['errors'] and summary['full_scored_method_panels']==2
    assert row['prediction_cases_submitted']==50 and row['scored_cases']==49
    assert row['APD_percent']==81 and row['closed_APD_percent']==90
    assert row['global_EPE_m']==.2 and row['closed_EPE_m']==.1
    assert row['new_minus_ours_pp']==1 and row['same_complete_scored_case_ids'] is True


def test_wrong_49_world_adt_exclusion_is_rejected(tmp_path):
    root=tmp_path/'eval_full_20261105';path=fixture(root);score=json.loads(path.read_text());score['scored_sample_ids']=score['sample_ids'][1:];save(path,score)
    summary=module.publish(root);row=new_row(summary)
    assert row['state']=='结果待核验' and row['APD_percent'] is None
    assert summary['full_scored_method_panels']==1 and 'new_minus_ours_pp' not in row
    assert 'exact official full eligibility' in summary['errors'][0]['error']


def test_other_world_dataset_missing_case_is_rejected(tmp_path):
    root=tmp_path/'eval_full_20261105';path=fixture(root,'trajectory_pstudio_mini');score=json.loads(path.read_text());score['scored_sample_ids'].pop();save(path,score)
    summary=module.publish(root);row=new_row(summary)
    assert row['APD_percent'] is None and row['state']=='结果待核验'
    assert summary['full_scored_method_panels']==1 and 'new_minus_ours_pp' not in row


def test_incomplete_world_metrics_do_not_leave_partial_APD_or_crash_pairing(tmp_path):
    root=tmp_path/'eval_full_20261105';path=fixture(root);score=json.loads(path.read_text());del score['metrics']['epe_sim3_closed'];save(path,score)
    summary=module.publish(root);row=new_row(summary)
    assert row['state']=='结果待核验' and row['APD_percent'] is None and row['APD_sample_std_pp'] is None
    assert 'closed_APD_percent' not in row and 'global_EPE_m' not in row and 'new_minus_ours_pp' not in row
    assert summary['full_scored_method_panels']==1 and len(summary['errors'])==1
