"""Build full-task coverage from independently validated native metadata."""
import collections
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]


def read(path):return json.loads(Path(path).read_text())
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def main():
    snapshot=read(HERE/'native_metadata_snapshot.json')
    old=WORK/'outputs/balanced_subset_20261104'
    base=read(old/'tap_mv_inventory.json')['datasets']
    panels={}
    for name,p in base.items():
        mf=snapshot['manifests'][p['canonical_manifest']]
        panels[name]={'panel_id':name,'benchmark':p['benchmark'],'canonical_manifest':mf['path'],
                      'canonical_manifest_sha256':mf['sha256'],'protocol':mf['protocol'],
                      'official_sources':mf['official_sources'],'input_condition':mf['input_condition'],
                      'coordinates':mf['coordinates'],'cases':mf['cases']}
    for panel in read(old/'world_geometry_panels.json')['panels']:
        source=next(m for m in panel['methods'] if m['method']=='Ours134400')
        mf=snapshot['manifests'][source['source_manifest']]
        cases=[c for c in mf['cases'] if c['subset']==panel['dataset']]
        panels[panel['panel_id']]={'panel_id':panel['panel_id'],'benchmark':'worldtrack','task':panel['task'],
            'canonical_manifest':mf['path'],'canonical_manifest_sha256':mf['sha256'],'protocol':mf['protocol'],
            'official_sources':mf['official_sources'],'input_condition':mf['input_condition'],
            'coordinates':mf['coordinates'],'cases':cases}
    methods=['Ours134400','ARROW','MoSE3','Point4D-direct-query','VGGT-Omega-1B-512+TrackEverything','UniQuery4R','ReLift+AllTracker+DA3']
    supported={m:[] for m in methods}
    supported['Ours134400']=list(panels)
    for m in ['ARROW','VGGT-Omega-1B-512+TrackEverything']:supported[m]=list(base)
    for m in ['MoSE3','Point4D-direct-query']:supported[m]=[k for k in base if base[k]['benchmark']=='tapvid3d']
    supported['UniQuery4R']=[k for k in panels if panels[k]['benchmark']=='worldtrack']
    supported['ReLift+AllTracker+DA3']=[k for k in panels if k.startswith('trajectory_')]
    rows=[]
    for method in methods:
        for name in supported[method]:
            panel=panels[name];by_id={c['id']:c for c in panel['cases']};groups={};rejected=[]
            for producer in snapshot['producers']:
                if producer['method']!=method:continue
                manifest=snapshot['manifests'][producer['source_manifest']]
                relevant=[c for c in manifest['cases'] if c['id'] in by_id]
                if not relevant:continue
                if any(manifest[k]!=panel[k] for k in ['benchmark','input_condition','coordinates','official_sources']):
                    rejected.append({'producer':producer['prediction_index'],'reason':'Producer protocol/official-source differs'});continue
                group=groups.setdefault(producer['group_signature_sha256'],{'signature_sha256':producer['group_signature_sha256'],
                    'signature_components':producer['signature_components'],'checkpoint_sha256':producer['checkpoint_sha256'],
                    'checkpoint_canonical_sha256':producer['checkpoint_canonical_sha256'],
                    'inference_settings':producer['inference_settings'],'runtime_versions':producer['runtime_versions'],
                    'numerics':producer['numerics'],'producers':[],'case_sources':{}})
                source={'prediction_index':producer['prediction_index'],'index_exists':producer['index_exists'],
                    'prediction_index_sha256':producer.get('prediction_index_sha256'),'run_state':producer['run_state'],
                    'run_state_file_sha256':producer['run_state_file_sha256'],'run_state_canonical_sha256':producer['run_state_canonical_sha256'],
                    'manifest':producer['source_manifest'],'manifest_sha256':producer['source_manifest_sha256'],
                    'eligible_case_ids':[],'receipt_only':not producer['index_exists']}
                if method=='Ours134400':
                    assert producer['loader_expected_step']==134400
                    assert producer['checkpoint_step'] in [None,134400]
                    assert producer['checkpoint_sha256']=='2557796c0068dbf33550ea4098d12b5f779cfab1e977decaf8ba13b79feaccb0'
                for case in producer['cases']:
                    cid=case['id']
                    if cid not in by_id:continue
                    expected=by_id[cid]
                    if any(case[k]!=expected[k] for k in ['input_sha256','frames','queries']):
                        rejected.append({'producer':producer['prediction_index'],'id':cid,'reason':'Canonical inputSHA/frames/queries differs'});continue
                    source['eligible_case_ids'].append(cid)
                    group['case_sources'].setdefault(cid,[]).append(case)
                group['producers'].append(source)
            preferred=max(groups.values(),key=lambda g:len(g['case_sources'])) if groups else None
            existing=set(preferred['case_sources']) if preferred else set()
            templates=[]
            for plan in snapshot['plans']:
                for job in plan['jobs']:
                    if job['method']!=method or not job['manifest']:continue
                    mf=snapshot['manifests'].get(job['manifest'])
                    if not mf:continue
                    originals={c['id']:c for c in mf['cases']}
                    if not set(by_id)<=set(originals):continue
                    if any(originals[cid]['input_sha256']!=by_id[cid]['input_sha256'] for cid in by_id):continue
                    if preferred:
                        if job['expected_checkpoint_sha256']!=preferred['checkpoint_sha256']:continue
                        if method!='Ours134400' and job['expected_checkpoint_canonical_sha256']!=preferred['checkpoint_canonical_sha256']:continue
                        adapter=(preferred['inference_settings'] or {}).get('adapter_sha256')
                        if adapter and adapter not in [a['sha256'] for a in job['checkpoint_artifacts'] or []]:continue
                    templates.append(dict(job,source_plan=plan['path'],source_plan_sha256=plan['sha256'],original_manifest_sha256=mf['sha256'],original_job_case_count=len(mf['cases'])))
            templates.sort(key=lambda t:(('integration_recovery_20261104' not in t['source_plan']),('formal_plan.json' not in t['source_plan']),t['source_plan']))
            assert templates,(method,name,'No real original full-task template')
            missing=[c['id'] for c in panel['cases'] if c['id'] not in existing]
            row={'id':method.replace('+','_').replace(' ','_')+'__'+name,'method':method,'panel_id':name,
                'total':len(by_id),'existing':len(existing),'missing':len(missing),
                'existing_ids':[c['id'] for c in panel['cases'] if c['id'] in existing],'missing_ids':missing,
                'preferred_group_signature_sha256':preferred['signature_sha256'] if preferred else None,
                'groups':list(groups.values()),'incompatible_groups_not_merged':len(groups)>1,'rejected_sources':rejected,
                'source_original':templates[0],'source_original_alternatives':templates[1:],
                'missing_input_copy_bytes':sum(by_id[cid]['input_bytes'] for cid in missing),
                'reuse_mode':'Native index + actual receipts, or committed receipt-only source; never synthesize producer run-state/index',
                'full_prediction_cache_ready':not missing,'full_official_score_reuse_requires_original_full_score_binding':not missing}
            if method=='Ours134400' and name=='drivetrack_minival':
                row['source_unwrapped_physical_gpu_affinity']=4
                row['source_gpu_lock']='@@BENCHMARK_ROOT@@/evaluation/.gpu4_single_inference.lock'
                row['affinity_reason']='Frozen streamed wrapper line103 hardcodes GPU4 lock and line37 checks inherited FD inode; source wrapper is unchanged.'
                row['safe_routing_override']={'script':str(HERE/'route_ours_drive_lock.py'),
                    'sha256':hashlib.sha256((HERE/'route_ours_drive_lock.py').read_bytes()).hexdigest(),
                    'physical_gpu_affinity':[3,4],'changes':['inherited GPU lock guard only','one old diagnostic status literal only'],
                    'original_source_affinity_applies_only_without_independent_routing_shim':True}
            rows.append(row)
    totals={method:{k:sum(r[k] for r in rows if r['method']==method) for k in ['total','existing','missing','missing_input_copy_bytes']} for method in methods}
    missing_paths={c['input']:c['input_bytes'] for r in rows for c in panels[r['panel_id']]['cases'] if c['id'] in r['missing_ids']}
    data={'schema':'full-evaluation-cache-inventory-v1','captured_at_utc':snapshot['captured_at_utc'],
        'canonical_panels':panels,'rows':rows,'totals_by_method':totals,
        'total_missing_GPU_cases':sum(r['missing'] for r in rows),'total_unique_reusable_cases':sum(r['existing'] for r in rows),
        'metadata_snapshot':str(HERE/'native_metadata_snapshot.json'),'metadata_snapshot_sha256':hashlib.sha256((HERE/'native_metadata_snapshot.json').read_bytes()).hexdigest(),
        'signature_rule':'Entire actual run-state except manifest_sha256. CP/settings/source/runtime/numerics/device remain exact. Case reuse also requires exact canonical ID+inputSHA+frames+queries and real committed receipt.',
        'large_prediction_or_input_payloads_rehashed':False,'metadata_only':True,'GPU_started':False,'live_queue_modified':False,
        'canonical_input_copy_bytes_all_method_tasks':sum(r['missing_input_copy_bytes'] for r in rows),
        'unique_missing_input_bytes_if_shared_preparation':sum(missing_paths.values()),
        'full_scope_note':'Complete canonical dataset case lists, no fixed subset sampling. World trajectory prediction canonical200; evaluator retains official ADT49eligible convention. No invented cross-dataset Overall.',
        'ours_identity':{'step':134400,'sha256':'2557796c0068dbf33550ea4098d12b5f779cfab1e977decaf8ba13b79feaccb0'},
        'unsupported':'No new speculative adapters: UQ/ReLift exclude TAP arbitrary-time queries/MV; MoSE/Point exclude MV/World; ARROW/TE exclude World; all6 exclude historical geometry tasks.'}
    (HERE/'full_inventory.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    lines=['# 全量评估缓存盘点','',f"元数据快照：{snapshot['captured_at_utc']}。Ours固定step134400。所有案例按原canonical顺序；没有启动GPU或改队列。",'',
        '| 方法 | 全量案例 | 可复用 | 缺失需推理 |','|---|---:|---:|---:|']
    for method,t in totals.items():lines.append(f"| {method} | {t['total']} | {t['existing']} | {t['missing']} |")
    lines+=['','| 方法 | 任务 | 总数 | 已有 | 缺失 |','|---|---|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['method']} | {r['panel_id']} | {r['total']} | {r['existing']} | {r['missing']} |")
    lines+=['',f"合计还需{data['total_missing_GPU_cases']}例，其中Ours DriveTrack补30；其20条已提交receipt有效，最终index不存在不代表无缓存。",'',
        '优先补齐：Panoptic ARROW/TE各1例football；DexYCB TE5例；MV-Kubric ARROW/TE各25例。UQ TUM8、Point Drive50、ARROW Dex10已有完整预测，可仅CPU验证/重汇总。', '',
        'Ours Drive原始wrapper硬编码GPU4锁；独立已核验route_ours_drive_lock.py仅路由父GPU3/4锁并重定向一个诊断status字面量，因此新prepare不再有GPU4亲和限制，模型/producer身份不变。', '',
        '跨run只有除manifestSHA外整个run-state完全一致才同组；不同组没有混合。正式源plan/job原始argv、冻结SHA、每个缺失ID对应input路径/SHA/帧数/query数见JSON。所有47个producer均由真实index/runstate/逐例receipt绑定；未重新读取或hash大NPZ和权重。', '',
        f"若每方法单独复制缺失inputs约{data['canonical_input_copy_bytes_all_method_tasks']/2**30:.1f}GiB；共享准备可降至约{data['unique_missing_input_bytes_if_shared_preparation']/2**30:.1f}GiB，另需预测输出空间。",'',
        '全量评分仍需完整ID覆盖、同协议及原模型身份绑定；不能把部分新producer索引伪称原full producer。World预测保留canonical200，官方ADT49eligible约定不变，不合并各数据集排名。','']
    (HERE/'full_inventory.md').write_text('\n'.join(lines))
    print(json.dumps({'rows':len(rows),'panels':len(panels),'totals':totals,'all_missing':data['total_missing_GPU_cases'],'source_rejections':sum(len(r['rejected_sources']) for r in rows)}))


if __name__=='__main__':main()
