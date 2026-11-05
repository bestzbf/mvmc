"""Pure stdlib, read-only display extraction from existing score dictionaries.

Same-scope means remain the saved official values. A true subset uses only
existing per-case scalar metrics after complete-source mean regression. No model,
metric implementation, GT, RGB or prediction archive is imported or opened.
"""
import math
import statistics


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _aliases(panel):
    panel = panel.lower()
    if 'drivetrack' in panel: return ['drivetrack']
    if 'pstudio' in panel: return ['pstudio_mini', 'pstudio']
    if 'adt' in panel: return ['adt_mini', 'adt']
    if 'tum' in panel: return ['tum']
    if 'dexycb' in panel: return ['dexycb']
    if 'panoptic' in panel: return ['panoptic']
    if 'kubric' in panel: return ['mv-kubric', 'mv_kubric']
    if 'po_mini' in panel or 'reconstruction_po' in panel: return ['po_mini']
    if 'ds_mini' in panel: return ['ds_mini']
    return [panel]


def _node(metrics, keys, aliases, path=()):
    if not isinstance(metrics, dict): return None
    if any(key in metrics for key in keys): return metrics, path
    for alias in aliases:
        if alias in metrics:
            found = _node(metrics[alias], keys, [], path+(alias,))
            if found: return found
    found = [_node(value, keys, [], path+(key,)) for key,value in metrics.items() if isinstance(value, dict)]
    found = [value for value in found if value is not None]
    if len(found) > 1: raise ValueError('Ambiguous dataset/protocol metric branch')
    return found[0] if found else None


def _cached_mean(values, mv=False, reject_rounding_tie=False):
    mean = statistics.fmean(values)
    if not mv: return mean
    scaled = mean*100.0
    if reject_rounding_tie and abs((scaled-math.floor(scaled))-0.5) < 1e-9:
        raise ValueError('MV half-cent rounding boundary requires a pre-materialized official subset aggregate')
    return round(scaled)/100.0


def _saved_std(score, flatkey, official_mean):
    for group in ('metric_robustness', 'robust_diagnostics', 'statistics'):
        entry = score.get(group, {}).get(flatkey) if isinstance(score.get(group),dict) else None
        if not isinstance(entry, dict): continue
        stored_mean = entry.get('official_mean_preserved', entry.get('official_rounded_mean'))
        if _finite(stored_mean) and not math.isclose(stored_mean,official_mean,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('Saved statistic belongs to another official mean')
        for key in ('sample_std', 'sample_std_diagnostic', 'sample_std_ddof1'):
            if _finite(entry.get(key)):
                return entry[key], group+'.'+flatkey+'.'+key, entry.get('valid_n',entry.get('valid_N',entry.get('N')))
    std = score.get('sample_std_ddof1')
    if isinstance(std,dict) and _finite(std.get(flatkey)):
        return std[flatkey], 'sample_std_ddof1.'+flatkey, None
    return None, score.get('std_unavailable_reason','No admitted complete per-case/std cache'), None


def extract_native_metrics(score, benchmark, panel_id, target_ids=None):
    """Return canonical metric records; directions are ``max``/``min``.

    Unsupported/nonfinite metrics are omitted. Missing std stays None. The
    returned IDs and alignment are part of the comparison scope, not decoration.
    ``target_ids`` must be a unique ordered subsequence of the saved scope.
    """
    aliases = _aliases(panel_id)
    mv = benchmark == 'mvtracker'
    if benchmark == 'tapvid3d':
        definitions = [('APD_percent','average_pts_within_thresh',100.0,'max','%','median'),
            ('AJ_percent','average_jaccard',100.0,'max','%','median'),
            ('OA_percent','occlusion_accuracy',100.0,'max','%','median')]
    elif mv:
        unit = '100×scene units' if 'kubric' in panel_id.lower() else 'cm'
        alignment = score.get('protocol',{}).get('mode',score.get('input_condition','declared_original_protocol'))
        definitions = [(name,'3dpt/model__'+key+'__any',1.0,direction,metric_unit,alignment) for name,key,direction,metric_unit in (
            ('APD_percent','average_pts_within_thresh','max','%'),('AJ_percent','average_jaccard','max','%'),
            ('OA_percent','occlusion_accuracy','max','%'),('ATE_native','ate_visible','min',unit),
            ('MTE_native','mte_visible','min',unit),('FDE_native','fde_visible','min',unit),
            ('Survival_percent','survival','max','%'))]
    elif benchmark == 'worldtrack':
        definitions = [('APD_global_percent','avg_pts_global',100.0,'max','%','global_median_scale'),
            ('APD_closed_percent','avg_pts_sim3_closed',100.0,'max','%','closed_sim3'),
            ('EPE_global_m','epe_global',1.0,'min','m','global_median_scale'),
            ('EPE_closed_m','epe_sim3_closed',1.0,'min','m','closed_sim3')]
    else: raise ValueError('Unknown benchmark: '+benchmark)
    keys = [x[1] for x in definitions]
    metrics = score.get('metrics',score.get('official_mean_metrics',{}))
    found = _node(metrics,keys,aliases)
    if not found: return {}
    aggregate, branch = found
    rows = score.get('per_case',[])
    source_ids = list(score.get('sample_ids') or [row['id'] for row in rows])
    matching = [i for i in source_ids if i.split('/')[0] in aliases]
    if matching: source_ids = matching
    if not source_ids or len(set(source_ids)) != len(source_ids): raise ValueError('Missing/duplicate saved scope IDs')
    target = list(source_ids if target_ids is None else target_ids)
    if not target or len(set(target)) != len(target) or [i for i in source_ids if i in set(target)] != target:
        raise ValueError('Target IDs are missing, duplicated or reordered')
    scored = [i for i in score.get('scored_sample_ids',source_ids) if i in set(source_ids)]
    selected = [i for i in target if i in set(scored)]
    if not selected: raise ValueError('No officially eligible scored target cases')
    by_id = {}
    for row in rows:
        if row['id'] not in set(source_ids): continue
        if row['id'] in by_id: raise ValueError('Duplicate cached per-case ID')
        by_id[row['id']] = row
    same_scope = target == source_ids
    support = score.get('metric_support') or {}
    unavailable = set(score.get('not_computed_metrics') or [])
    result = {}
    for name,key,factor,direction,unit,alignment in definitions:
        if name in ('AJ_percent','OA_percent') and (support.get('native_visibility') is False or support.get('AJ' if name=='AJ_percent' else 'OA') is False or ('average_jaccard' if name=='AJ_percent' else 'occlusion_accuracy') in unavailable): continue
        official = aggregate.get(key)
        if not _finite(official): continue
        flatkey = ('median.'+key) if benchmark=='tapvid3d' else key
        values = {}
        for ident,row in by_id.items():
            node = _node(row.get('metrics',{}),[key],aliases)
            if node and _finite(node[0].get(key)): values[ident] = node[0][key]
        full_coverage = set(by_id) >= set(scored)
        finite_source = [values[i] for i in scored if i in values]
        regression = None
        if full_coverage and finite_source:
            reconstructed = _cached_mean(finite_source,mv)
            regression = math.isclose(reconstructed,official,rel_tol=1e-12,abs_tol=1e-12)
        if same_scope:
            mean = official
            std,std_source,valid_n = _saved_std(score,flatkey,official)
            if std is None and regression:
                std = statistics.stdev(finite_source) if len(finite_source)>1 else None
                std_source = 'complete cached per-case scalars; full aggregate regression passed'
            valid_n = valid_n if valid_n is not None else (len(finite_source) if full_coverage else len(scored))
            mean_source = 'saved official aggregate preserved'
        else:
            if not full_coverage or regression is not True: raise ValueError('Subset scalar reuse needs complete source per-case/full-mean regression: '+flatkey)
            chosen = [values[i] for i in selected if i in values]
            if not chosen: continue
            mean = _cached_mean(chosen,mv,True)
            std = statistics.stdev(chosen) if len(chosen)>1 else None
            std_source = 'exact selected cached scalar sample std (ddof1)'
            valid_n = len(chosen)
            mean_source = 'cached scalar equal-video subset mean; full source aggregate regression passed'
        result[name] = {'mean':mean*factor,'sample_std':std*factor if std is not None else None,
            'unit':unit,'std_unit':'percentage points' if unit=='%' else unit,'direction':direction,
            'rawkey':flatkey,'native_key':flatkey,'raw_mean':mean,'scale_factor':factor,'available':True,
            'rawpath':['metrics',*branch,key],'alignment':alignment,'valid_N':valid_n,'source_scope_N':len(source_ids),
            'mean_source':mean_source,'std_source':std_source,'sample_std_ddof':1,
            'scope_ids':target,'scored_scope_ids':selected,'source_scope_ids':source_ids,
            'source_full_aggregate_regression_pass':regression,'subset_of_source':not same_scope}
    return result
