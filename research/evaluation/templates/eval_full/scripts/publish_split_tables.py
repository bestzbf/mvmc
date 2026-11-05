"""Render matched subsets and full evaluations separately, from metadata only."""
import argparse
import json
import math
import os
from pathlib import Path

OURS = 'Ours134400'
COT = 'VGGT-Omega-1B-512+CoTracker3-Offline'
TAPIP = 'VGGT-Omega-1B-512+TAPIP3D'
MVT = 'VGGT-Omega-1B-512+MVTracker'
TE = 'VGGT-Omega-1B-512+TrackEverything'
RELIFT = 'ReLift+AllTracker+DA3'
NEW = {'ARROW', 'MoSE3', 'Point4D-direct-query', TE, 'UniQuery4R', RELIFT}
LABELS = {COT: 'VGGT + CoTracker3', TAPIP: 'VGGT + TAPIP3D', MVT: 'VGGT + MVTracker',
          TE: 'TrackEverything + VGGT', 'Point4D-direct-query': 'Point4D',
          RELIFT: 'ReLift + AllTracker + DA3'}
PANELS = {
    'adt_minival': ('ADT', 5), 'drivetrack_minival': ('DriveTrack 固定', 10),
    'pstudio_minival': ('PStudio', 5), 'drivetrack_first20': ('DriveTrack 独立', 20),
    'dexycb': ('DexYCB', 5), 'panoptic': ('Panoptic', 5), 'mv-kubric': ('MV-Kubric', 5),
    'trajectory_adt_mini': ('ADT', 5), 'trajectory_ds_mini': ('Dynamic Replica', 5),
    'trajectory_po_mini': ('PointOdyssey', 5), 'trajectory_pstudio_mini': ('PStudio', 5),
    'reconstruction_tum': ('TUM', 5), 'reconstruction_po_mini': ('PointOdyssey', 5),
}
FAMILIES = [
    ('TAPVid-3D', ['adt_minival', 'drivetrack_minival', 'pstudio_minival', 'drivetrack_first20'],
     [OURS, COT, TAPIP, 'ARROW', 'MoSE3', 'Point4D-direct-query', TE]),
    ('同步多相机跟踪', ['dexycb', 'panoptic', 'mv-kubric'], [OURS, MVT, COT, TAPIP, 'OmniX', 'ARROW', TE]),
    ('WorldTrack轨迹 · global median',
     ['trajectory_adt_mini', 'trajectory_ds_mini', 'trajectory_po_mini', 'trajectory_pstudio_mini'],
     [OURS, '4RC', 'St4RTrack', 'V-DPM', 'OmniX', 'OpenD4RT', 'UniQuery4R', RELIFT]),
    ('WorldTrack重建 · global median', ['reconstruction_tum', 'reconstruction_po_mini'],
     [OURS, '4RC', 'St4RTrack', 'V-DPM', 'OmniX', 'OpenD4RT', 'Any4D', 'UniQuery4R']),
]
METRICS = {
    'AJ_percent': ('AJ₃D', '%', 'max'),
    'OA_percent': ('OA', '%', 'max'),
    'ATE_native': ('ATE_visible', '原生位置单位，见列', 'min'),
    'MTE_native': ('MTE_visible', '原生位置单位，见列', 'min'),
    'FDE_native': ('FDE_visible', '原生位置单位，见列', 'min'),
    'Survival_percent': ('Survival', '%', 'max'),
    'APD_closed_percent': ('APD₃D · closed-form Sim3', '%', 'max'),
    'EPE_global_m': ('EPE · global median', 'm', 'min'),
    'EPE_closed_m': ('EPE · closed-form Sim3', 'm', 'min'),
}
EXTRA_METRICS = [
    ['AJ_percent', 'OA_percent'],
    ['AJ_percent', 'OA_percent', 'ATE_native', 'MTE_native', 'FDE_native', 'Survival_percent'],
    ['APD_closed_percent', 'EPE_global_m', 'EPE_closed_m'],
    ['APD_closed_percent', 'EPE_global_m', 'EPE_closed_m'],
]


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(value)
    os.replace(tmp, path)


def cell(mean, std, best=False, digits=2):
    value = f'**{mean:.{digits}f}**' if best else f'{mean:.{digits}f}'
    return value + (f' ± {std:.{digits}f}' if std is not None else '')


def scope(panel, ids):
    # Trajectory and reconstruction may share IDs; their task stays distinct.
    assert len(ids) == len(set(ids))
    return panel, tuple(sorted(ids))


def insert(group, method, mean, std):
    if mean is not None:
        assert math.isfinite(mean) and (std is None or math.isfinite(std) and std >= 0)
    old = group['methods'].get(method)
    if old is not None:
        assert old['mean'] is not None and mean is not None
        assert math.isclose(old['mean'], mean, abs_tol=1e-8)
        if old['std'] is not None and std is not None:
            assert math.isclose(old['std'], std, abs_tol=1e-8)
        return
    group['methods'][method] = {'mean': mean, 'std': std}


def subset_groups(subset, legacy):
    groups, primary = {}, {}

    def obtain(panel, ids, label):
        key = scope(panel, ids)
        if key not in groups:
            groups[key] = {'panel': panel, 'ids': list(ids), 'N': len(ids), 'label': label, 'methods': {}}
        return key, groups[key]

    for pair in subset['paired_comparisons']:
        assert pair['same_case_set_verified'] is True and pair['ours_checkpoint_step'] == 134400
        assert len(pair['sample_ids']) == pair['N']
        key, group = obtain(pair['dataset'], pair['sample_ids'], pair['dataset_label'])
        assert pair['dataset'] not in primary or primary[pair['dataset']] == key
        primary[pair['dataset']] = key
        metric = pair['metrics']['APD_global_percent' if pair['benchmark'] == 'worldtrack' else 'APD_percent']
        for method, name in [(OURS, 'ours'), (pair['method'], 'new_model')]:
            insert(group, method, metric[name]['official_mean'], metric[name]['sample_std'])
    for row in legacy['rows']:
        assert row['case_IDs_exactly_match_ours'] is True
        assert len(row['sample_ids']) == row['N']
        _, group = obtain(row['dataset'], row['sample_ids'], row['label'])
        insert(group, OURS, row['ours_APD_percent'], row.get('ours_sample_std_pp'))
        insert(group, row['method'], row['APD_percent'], row.get('sample_std_pp'))
    for row in subset['incomplete_supported_subsets']:
        group = groups[primary[row['dataset']]]
        assert group['N'] == row['N']
        insert(group, row['method'], None, None)
    current = {panel: groups[key] for panel, key in primary.items()}
    historical = [g for key, g in groups.items() if key not in set(primary.values())]
    return current, historical


def method_label(method):
    name = LABELS.get(method, method)
    return name + ('（新）' if method in NEW else '')


def metric_value(value, full, metric):
    if metric is None:
        return (value['APD_percent'], value.get('APD_sample_std_pp')) if full else (value['mean'], value['std'])
    item = value.get('display_metrics', {}).get(metric)
    return (None, None) if item is None else (item['mean'], item.get('sample_std'))


def matrix(lines, panels, methods, values, full=False, metric=None):
    name, unit, direction = METRICS[metric] if metric else ('APD₃D', '%', 'max')
    arrow = '↑' if direction == 'max' else '↓'
    headers = []
    for panel in panels:
        label, n = PANELS[panel]
        if full:
            label = 'DriveTrack' if panel == 'drivetrack_minival' else label
            n = next(iter(values[panel].values()))['prediction_cases_total']
        column = label + ('（评分49 / 预测50）' if full and panel == 'trajectory_adt_mini' else f'（{n}例）')
        if metric in ['ATE_native', 'MTE_native', 'FDE_native']:
            column += ' / ' + ('100×scene-unit' if panel == 'mv-kubric' else 'cm')
        headers.append(column)
    lines += [f'| 方法 / {name} {arrow} ({unit}) | ' + ' | '.join(headers) + ' |', '| --- | ' + ' | '.join('---:' for _ in panels) + ' |']
    best_by_panel = {}
    for panel in panels:
        means = [metric_value(values[panel][m], full, metric)[0] for m in methods]
        valid = [v for v in means if v is not None]
        best_by_panel[panel] = (max(valid) if direction == 'max' else min(valid)) if valid else None
    complete = 0
    for method in methods:
        fields = [method_label(method)]
        for panel in panels:
            value = values[panel][method]
            mean, std = metric_value(value, full, metric)
            if mean is None:
                primary = value['APD_percent'] if full else value['mean']
                formatted = ('未提供' if metric and primary is not None else
                             f'待评分（{value["prediction_cases_submitted"]}/{value["prediction_cases_total"]}）' if full else '未完成')
            else:
                complete += 1
                formatted = cell(mean, std,
                                 math.isclose(mean, best_by_panel[panel], rel_tol=0, abs_tol=1e-9),
                                 digits=4 if metric in ['EPE_global_m', 'EPE_closed_m'] else 2)
            fields.append(formatted)
        lines.append('| ' + ' | '.join(fields) + ' |')
    return complete, len(methods) * len(panels)


def extra_tables(lines, panels, methods, values, keys, full=False):
    available = [key for key in keys if any(metric_value(values[p][m], full, key)[0] is not None
                                           for p in panels for m in methods)]
    if not available:
        return
    lines += ['<details>', '<summary>其他指标：' + '、'.join(METRICS[k][0] for k in available) + '</summary>', '']
    for key in available:
        supported = [m for m in methods if any(metric_value(values[p][m], full, key)[0] is not None for p in panels)]
        name, unit, direction = METRICS[key]
        lines += ['### ' + name + (' ↑' if direction == 'max' else ' ↓') + f'（{unit}）', '']
        matrix(lines, panels, supported, values, full=full, metric=key)
        omitted = [LABELS.get(m, m) for m in methods if m not in supported]
        if omitted:
            lines += ['', '当前接入未提供该指标或尚未评分：' + '、'.join(omitted) + '。']
        lines.append('')
    lines += ['</details>', '']


def attach_subset_metrics(current, historical, root):
    groups = list(current.values()) + historical
    lookup = {(g['panel'], tuple(sorted(g['ids'])), m): v
              for g in groups for m, v in g['methods'].items()}
    loaded = 0
    for name in ['legacy_metrics_extended.json', 'new_metrics_extended.json']:
        path = root / 'planning' / name
        if not path.exists():
            continue
        for row in read(path)['rows']:
            panel = row.get('panel_id', row.get('dataset'))
            key = (panel, tuple(sorted(row['sample_ids'])), row['method'])
            if key not in lookup:
                continue
            value = lookup[key]
            for metric, item in row['metrics'].items():
                if metric not in METRICS or item.get('mean') is None:
                    continue
                existing = value.setdefault('display_metrics', {}).get(metric)
                if existing:
                    assert math.isclose(existing['mean'], item['mean'], rel_tol=0, abs_tol=1e-7)
                    if existing.get('sample_std') is not None and item.get('sample_std') is not None:
                        assert math.isclose(existing['sample_std'], item['sample_std'], rel_tol=0, abs_tol=1e-7)
                    elif existing.get('sample_std') is None and item.get('sample_std') is not None:
                        existing['sample_std'] = item['sample_std']
                else:
                    value['display_metrics'][metric] = item
                    loaded += 1
    return loaded


def publish(base):
    full_root = base / 'eval_full_20261105'
    subset = read(base / 'eval_10h_20261104/新模型同子集对比.json')
    full = read(full_root / 'full_results.json')
    legacy = read(full_root / 'planning/legacy_subset_rows.json')
    current, historical = subset_groups(subset, legacy)
    extra_count = attach_subset_metrics(current, historical, full_root)
    destination = base / '评估对比'
    nav = '[子集评估表](子集评估表.md) · [全量评估表](全量评估表.md)'
    coverage, missing = [], []
    for title, panels, methods in FAMILIES:
        for panel in panels:
            assert set(current[panel]['methods']) == set(methods), 'Unexpected/missing method in ' + panel
            for method, value in current[panel]['methods'].items():
                if value['mean'] is None:
                    missing.append({'panel': panel, 'method': method, 'N': current[panel]['N']})
        complete = sum(current[p]['methods'][m]['mean'] is not None for p in panels for m in methods)
        coverage.append({'protocol': title, 'complete': complete, 'expected': len(panels)*len(methods)})
    done = sum(v['complete'] for v in coverage)
    expected = sum(v['expected'] for v in coverage)
    paired = len(subset['paired_comparisons'])
    pair_total = paired + len(subset['incomplete_supported_subsets'])
    pending_runtime = None
    queue_path = full_root / 'progress.json'
    if missing and queue_path.exists():
        queue = read(queue_path)
        mose = [r for r in queue['records'] if r['kind'] == 'gpu'
                and r['method'] == 'MoSE3' and r['dataset'] == 'adt_minival']
        committed = {case for r in mose for case in r.get('receipt_progress', {}).get('ids', [])}
        ready = len(committed & set(current['adt_minival']['ids']))
        if ready == current['adt_minival']['N']:
            state = '固定子集预测已齐，子集评分待汇总'
        elif any(r['state'] == 'running' for r in mose):
            state = 'ADT全量正在运行'
        elif mose and all(r['state'] == 'pending' for r in mose):
            state = 'ADT全量任务尚未开始'
        else:
            state = '固定子集仍有缺项，详见逐项队列'
        pending_runtime = {'updated_at_utc': queue['updated_at_utc'], 'fixed_subset_predictions_ready': ready,
                           'fixed_subset_predictions_total': current['adt_minival']['N'], 'state': state}
    lines = ['# 子集评估表', '', nav, '',
             f'已接入范围内：新模型配对 {paired}/{pair_total} 组；含Ours和旧基线的当前子集成绩 {done}/{expected} 项。', '',
             'Ours固定step134400。各主表为 APD₃D ↑ (%)，数值为均值 ± 跨案例样本标准差。↑越高越好，↓越低越好。按未舍入均值判断，每列仅最优均值加粗，并列最优同时加粗；标准差不参与排名。百分数保留两位小数，EPE（米）保留四位；未提供标准差时仅列均值。（新）标记本轮新增模型。', '',
             'APD₃D：平均阈值内3D点比例；AJ₃D：平均3D Jaccard；OA：遮挡判断准确率；EPE：端点误差。已提供的其他指标在各主表下方展开，名称、单位及优化方向分别标明。“未提供”不计为0或任务未完成。', '']
    if missing:
        lines += ['未完成：MoSE3 / ADT / TAP 固定5例。旧任务两次90分钟超时，没有完成预测。旧评估窗口已结束，这项子集仍未成功。', '']
        if pending_runtime:
            lines += [f'最新核验UTC {pending_runtime["updated_at_utc"]}：{pending_runtime["state"]}；固定5例已有有效预测 {pending_runtime["fixed_subset_predictions_ready"]}/5。', '']
    lines += ['原先8个“—”是当前适配未接入的组合：MoSE3、Point4D的3组多相机任务，以及ReLift的2组重建任务。它们不计入完成数，也不列入对应表；实际缺项单独标“未完成”。', '']
    for (title, panels, methods), count, keys in zip(FAMILIES, coverage, EXTRA_METRICS):
        lines += ['## ' + title + f' · APD₃D ↑ (%)（{count["complete"]}/{count["expected"]}项）', '']
        values = {p: current[p]['methods'] for p in panels}
        matrix(lines, panels, methods, values)
        lines.append('')
        extra_tables(lines, panels, methods, values, keys)
    lines += ['每一列内严格使用相同任务、相同案例ID。DriveTrack固定10例与独立20例分列；WorldTrack轨迹和重建即使案例名相同也分开。VGGT系列沿用Omega-1B-512配置。', '',
              'WorldTrack主表使用global median对齐；closed-form Sim3和EPE已在对应的可展开指标表中分别列出。新模型逐例胜负见[完整配对明细](../eval_10h_20261104/新模型同子集对比.md)。', '']
    if historical:
        lines += ['<details>', '<summary>历史3例结果（独立保留，不计入上方完成率）</summary>', '']
        for group in historical:
            lines += ['### ' + group['label'], '', '| 方法 | APD% ± 样本标准差 | Δ vs Ours（百分点） |', '| --- | ---: | ---: |']
            ours = group['methods'][OURS]['mean']
            best = max(v['mean'] for v in group['methods'].values())
            order = [OURS] + sorted(m for m in group['methods'] if m != OURS)
            for method in order:
                v = group['methods'][method]
                lines.append('| ' + ' | '.join([method_label(method), cell(v['mean'], v['std'], math.isclose(v['mean'], best, rel_tol=0, abs_tol=1e-9)), f'{v["mean"]-ours:+.2f}']) + ' |')
            lines.append('')
        lines += ['</details>', '']
    lines += [f'子集分数来源快照：{subset["snapshot_at_utc"]}。', '',
              '[新模型原始记录](../eval_10h_20261104/新模型同子集对比.json) · [旧基线原始记录](../eval_full_20261105/planning/legacy_subset_rows.json) · [当前覆盖清单](子集覆盖清单.json)', '',
              '[原共同子集报告](../balanced_subset_20261104/按数据集复用对比.md)另保留Ours126900历史几何结果。', '']
    save(destination / '子集评估表.md', '\n'.join(lines))
    audit = {'schema': 'matched-subset-display-v2', 'completed_results': done, 'expected_results': expected,
             'new_model_completed_pairs': paired, 'new_model_expected_pairs': pair_total,
             'protocols': coverage, 'missing': missing, 'current_groups': len(current),
             'historical_groups': len(historical), 'scope_key': 'task/panel + exact case ID set',
             'missing_runtime': pending_runtime,
             'additional_metric_records': extra_count,
             'score_metrics_recomputed': False}
    save(destination / '子集覆盖清单.json', json.dumps(audit, ensure_ascii=False, indent=2)+'\n')

    full_values = {}
    for row in full['rows']:
        if 'display_metrics' not in row and (full_root / row.get('score_path', '/nonexistent')).is_file():
            from native_display_metrics import extract_native_metrics
            row['display_metrics'] = extract_native_metrics(read(full_root / row['score_path']), row['benchmark'], row['panel_id'])
        full_values.setdefault(row['panel_id'], {})[row['method']] = row
    lines = ['# 全量评估表', '', nav, '',
             f'全量评分快照：{full["updated_at_utc"]}。完整评分{full["full_scored_method_panels"]}/{full["target_method_panels"]}组；Ours固定step134400。', '',
             '主指标 APD₃D ↑ (%)。本表范围为Ours及6个新模型的全量计划；数值为完整评分均值 ± 样本标准差。每列仅在已有完整评分中将最优均值加粗，标准差不参与排名；未齐全时不代表最终排名。“待评分（已提交预测/全量预测）”表示缺项进度。', '']
    for (title, all_panels, all_methods), keys in zip(FAMILIES, EXTRA_METRICS):
        panels = [p for p in all_panels if p != 'drivetrack_first20']
        methods = [m for m in all_methods if m == OURS or m in NEW]
        for panel in panels:
            assert set(full_values[panel]) == set(methods)
        lines += ['## ' + title + ' · APD₃D ↑ (%)', '']
        matrix(lines, panels, methods, full_values, full=True)
        lines.append('')
        extra_tables(lines, panels, methods, full_values, keys, full=True)
    lines += ['World ADT生成50例预测，按原官方统一规则评分49例，排除唯一首帧无可见点案例。其余World轨迹各50例。', '',
              '[全量明细（含Δ、Closed APD及EPE）](../eval_full_20261105/全量评估对比.md) · [运行进度](../eval_full_20261105/实时进度.md) · [全量安排](../eval_full_20261105/全量评估安排.md)', '']
    save(destination / '全量评估表.md', '\n'.join(lines))
    save(destination / '评估结果.md', '\n'.join([
        '# 评估结果', '', '| 查看内容 | 入口 |', '| --- | --- |',
        '| 固定子集上的新旧方法成绩及真实缺项 | [子集评估表](子集评估表.md) |',
        '| Ours与新模型的完整案例成绩及补齐进度 | [全量评估表](全量评估表.md) |', '',
        '子集和全量独立展示；队列结束不等于每项评估成功。', '']))
    return {'subset_comparisons': paired, 'subset_results_complete': done, 'subset_results_expected': expected,
            'full_scored_method_panels': full['full_scored_method_panels'], 'destination': str(destination)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.base), ensure_ascii=False))
