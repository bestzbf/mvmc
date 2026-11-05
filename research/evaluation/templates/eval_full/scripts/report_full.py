"""Continuously publish full-coverage scores; partial subsets never fill full means."""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import time
from publish_split_tables import publish as publish_split_tables
from native_display_metrics import extract_native_metrics


def read(p):return json.loads(Path(p).read_text())
def save(p,value):
    text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    t=p.with_name(p.name+'.tmp');t.write_text(text);os.replace(t,p)
def flatten(o,prefix=''):
    result={}
    for key,value in o.items():
        name=prefix+'.'+key if prefix else key
        if isinstance(value,dict):result.update(flatten(value,name))
        elif isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):result[name]=float(value)
    return result
def metric(values,suffix):
    matches=[v for k,v in flatten(values).items() if k==suffix or k.endswith('.'+suffix)]
    if len(matches)!=1:raise ValueError('Missing/ambiguous metric '+suffix)
    return matches[0]
def std(score,key):
    for field in ['sample_std_ddof1','metric_robustness','robust_diagnostics','statistics']:
        d=score.get(field)
        if not isinstance(d,dict):continue
        if key in d:
            v=d[key]
            if isinstance(v,(int,float)):return float(v)
            if isinstance(v,dict):
                for n in ['sample_std','sample_std_diagnostic','sample_std_ddof1']:
                    if isinstance(v.get(n),(int,float)):return float(v[n])
    return None
def scored_ids(score):return score.get('scored_sample_ids',score['sample_ids'])


def publish(root):
    inventory=read(root/'planning/full_inventory.json');targets=read(root/'full_targets.json')
    progress=read(root/'progress.json');bytarget={r['row_id']:r for r in targets['rows']}
    bypair={(r['method'],r['panel_id']):r['id'] for r in inventory['rows']}
    additions={k:set() for k in bytarget}
    for record in progress['records']:
        key=bypair.get((record['method'],record['dataset']))
        if key and record['kind']=='gpu':additions[key].update(record.get('receipt_progress',{}).get('ids',[]))
    rows=[];errors=[];scores={}
    for original in inventory['rows']:
        rid=original['id'];target=bytarget[rid];panel=inventory['canonical_panels'][original['panel_id']]
        complete_ids=set(original['existing_ids'])|additions[rid]
        row=dict(row_id=rid,method=original['method'],panel_id=original['panel_id'],benchmark=panel['benchmark'],
                 prediction_cases_submitted=len(complete_ids),prediction_cases_total=original['total'],
                 new_prediction_cases_submitted=len(additions[rid]-set(original['existing_ids'])),
                 state='等待全量预测/评分',APD_percent=None,APD_sample_std_pp=None)
        path=Path(target.get('score_path','/nonexistent'))
        if path.is_file():
            try:
                score=read(path)
                if target.get('score_sha256'):assert hashlib.sha256(path.read_bytes()).hexdigest()==target['score_sha256']
                assert score['full_frozen_coverage_verified'] is True
                ids=scored_ids(score);predids=score['sample_ids']
                assert predids==[c['id'] for c in panel['cases']] and len(set(ids))==len(ids)
                expected_scored=[i for i in predids if not (
                    original['panel_id']=='trajectory_adt_mini' and
                    i=='@@BENCHMARK_CASE_4BC7D2E490@@')]
                assert ids==expected_scored, 'Scored case IDs differ from exact official full eligibility'
                kind=panel['benchmark'];key=('median.average_pts_within_thresh' if kind=='tapvid3d' else
                    '3dpt/model__average_pts_within_thresh__any' if kind=='mvtracker' else 'avg_pts_global')
                factor=1 if kind=='mvtracker' else 100
                value=metric(score['metrics'],key)*factor;sd=std(score,key)
                score_fields=dict(APD_percent=value,APD_sample_std_pp=sd*factor if sd is not None else None,
                           scored_cases=len(ids),state='全量已评分（复用）' if target.get('reused_full_score') else '全量已评分',
                           score_path=str(path.relative_to(root)),prediction_cases_submitted=original['total'])
                if kind=='worldtrack':
                    score_fields.update(closed_APD_percent=metric(score['metrics'],'avg_pts_sim3_closed')*100,
                               global_EPE_m=metric(score['metrics'],'epe_global'),closed_EPE_m=metric(score['metrics'],'epe_sim3_closed'))
                score_fields['display_metrics']=extract_native_metrics(score,kind,original['panel_id'])
                row.update(score_fields)
                scores[rid]=score
            except (AssertionError,ValueError,KeyError) as exc:
                row['state']='结果待核验';errors.append(dict(row=rid,error=str(exc)))
        rows.append(row)
    ours={r['panel_id']:r for r in rows if r['method']=='Ours134400' and r['APD_percent'] is not None}
    for row in rows:
        reference=ours.get(row['panel_id'])
        if row['method']=='Ours134400' or row['APD_percent'] is None or reference is None:continue
        a,b=scores[row['row_id']],scores[reference['row_id']]
        if scored_ids(a)==scored_ids(b):
            row['ours_APD_percent_same_full_scope']=reference['APD_percent']
            row['new_minus_ours_pp']=row['APD_percent']-reference['APD_percent']
            row['same_complete_scored_case_ids']=True
        else:errors.append(dict(row=row['row_id'],error='Full comparison IDs differ; no paired delta shown'))
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    summary=dict(updated_at_utc=now,queue_updated_at_utc=progress['updated_at_utc'],
                 global_deadline=None,overall_status=progress['overall_status'],final=progress['final'],
                 full_scored_method_panels=len(scores),target_method_panels=40,
                 planned_new_prediction_cases=inventory['total_missing_GPU_cases'],
                 new_prediction_cases_submitted=sum(r['new_prediction_cases_submitted'] for r in rows),
                 rows=rows,errors=errors,queue_counts=progress['counts'],ours_checkpoint_step=134400,
                 full_scope='Original frozen manifests: TAP50 per split, MV10/6/30, World4x50 predictions with original eligible199, reconstruction TUM8/PO50')
    lines=['# 全量评估结果与进度','',
           '[子集快速表](../评估对比/子集评估表.md) · [全量快速表](../评估对比/全量评估表.md)','',
           f'更新UTC：{now}。无全局10小时时限；Ours固定step134400。','',
           f'全量已评分 {len(scores)}/40 个方法任务；新增预测已提交 {summary["new_prediction_cases_submitted"]}/{summary["planned_new_prediction_cases"]} 例。', '',
           '仅在完整同案例集合上显示配对差值。全量未齐时留空，不使用先前5/10/20例均值代替。APD单位%，样本标准差单位百分点；World主列为global-median，closed-form另列。', '',
           'WorldTrack ADT仍预测50例，但原官方规则排除1例首帧无可见点，评分分母49；其余World各50。此排除对所有方法相同。', '',
           'GPU仅3/4；GPU4有外来进程时跳过，释放后自动加入。训练不改。单项失败保留并继续其他任务；全部目标成功才报告成功，失败缺项将列needs_attention。']
    for pid,panel in inventory['canonical_panels'].items():
        lines+=['','## '+pid,'']
        if panel['benchmark']=='worldtrack':
            lines+=['| 方法 | 预测提交/全量 | 评分N | Global APD% | sample std | Closed APD% | Global EPE m | Closed EPE m | ΔAPD vs Ours | 状态 |',
                    '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
        else:
            lines+=['| 方法 | 预测提交/全量 | 评分N | APD% | sample std | ΔAPD vs Ours | 状态 |',
                    '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
        fmt=lambda v:'—' if v is None else f'{v:.4f}'
        for row in sorted([r for r in rows if r['panel_id']==pid],key=lambda r:(r['method']!='Ours134400',r['method'])):
            prefix=f'| {row["method"]} | {row["prediction_cases_submitted"]}/{row["prediction_cases_total"]} | {row.get("scored_cases","—")} | {fmt(row["APD_percent"])} | {fmt(row["APD_sample_std_pp"])} | '
            if panel['benchmark']=='worldtrack':prefix+=' | '.join(fmt(row.get(k)) for k in ['closed_APD_percent','global_EPE_m','closed_EPE_m'])+' | '
            lines.append(prefix+fmt(row.get('new_minus_ours_pp'))+f' | {row["state"]} |')
    lines+=['','[逐项队列](实时进度.md) · [全量结果JSON](full_results.json) · [运行进度JSON](progress.json)','',
            '以上“全量”指冻结计划内的全部案例；RGB-only query-anchor Sim3、ReLift64帧等原方法扩展说明继续保留，不增加论文原协议或formal admission声明。']
    save(root/'full_results.json',summary);save(root/'全量评估对比.md','\n'.join(lines)+'\n')
    publish_split_tables(root.parent)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--once',action='store_true');args=parser.parse_args();root=Path(args.root)
    while True:
        try:
            summary=publish(root)
            if args.once:
                print(json.dumps({k:v for k,v in summary.items() if k not in ['rows']},ensure_ascii=False));break
            if summary['final']:break
        except Exception as exc:
            save(root/'full_report_error.json',{'at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'error':repr(exc)})
            if args.once:raise
        time.sleep(30)
