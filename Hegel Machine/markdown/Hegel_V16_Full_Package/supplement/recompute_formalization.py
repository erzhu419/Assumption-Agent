#!/usr/bin/env python3
"""Audit the supplied Phase 3 report, optionally rebuilding kernels from cache.

No model or network calls. A missing cached action distribution stops the kernel
rebuild. The report-only audit uses stdlib; --sources additionally needs numpy.
Archived source functions are extracted with AST; their module-level imports,
clients, main routines, file writing, and paid classifiers are never executed.
Full graph edit distances and Blackwell LPs are NOT rerun.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
from itertools import combinations, product
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace
from typing import Dict, List, Tuple


ACCEPTED = {'iso', 'weak_iso', 'iso_strong', 'iso_latent', 'structural_match_only'}


def pair(row):
    return tuple(sorted((row['strategy_a'], row['strategy_b'])))


def accepted(label):
    return label in ACCEPTED or label.startswith('subsume')


def classify(row, fisher_thr, spec_thr):
    f, spectral, asym = row['fisher'], row['spectral'], row['kl_log_asym']
    confidence = max(0.0, 1.0 - f / max(fisher_thr, 1e-6))
    if f < fisher_thr * .7 and spectral < spec_thr and abs(asym) < .3:
        return 'iso', confidence
    if f < fisher_thr and abs(asym) < .3:
        return 'weak_iso', confidence * .7
    if f < fisher_thr:
        return ('subsume_b_over_a' if asym < 0 else 'subsume_a_over_b'), confidence * .6
    return 'distinct', 1.0-confidence


def combine(label, seg, blackwell):
    # run_formalization_v2.py 88–96, including elif precedence.
    if label == 'iso' and seg > .8:
        return 'iso_surface_only'
    if label == 'distinct' and seg < .3:
        return 'structural_match_only'
    if label in ('iso', 'weak_iso') and blackwell == 'both_dominate':
        return 'iso_strong'
    if label == 'distinct' and blackwell == 'both_dominate':
        return 'iso_latent'
    return label


def calibrate(rows):
    fisher = [r['fisher'] for r in rows]
    spectral = [r['spectral'] for r in rows]
    return (statistics.mean(fisher)-.5*statistics.pstdev(fisher),
            statistics.mean(spectral)+.2*statistics.pstdev(spectral))


def describe(values):
    return {'n': len(values), 'minimum': min(values), 'maximum': max(values),
            'mean': statistics.mean(values), 'population_sd': statistics.pstdev(values),
            'median': statistics.median(values)}


def extract(source_path, env, function_names=(), assignment_names=()):
    parsed = ast.parse(source_path.read_text(encoding='utf-8'), filename=str(source_path))
    body = []
    for node in parsed.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in function_names:
            body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {t.id for t in targets if isinstance(t, ast.Name)}
            if names & set(assignment_names):
                body.append(node)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source_path), 'exec'), env)


def rebuild_kernels(source_dir, rows):
    import numpy as np
    env = {'np': np, 'Dict': Dict, 'List': List, 'Tuple': Tuple, 'product': product,
           'Path': Path, 'json': json, '__name__': 'phase3_cached_audit'}
    def src(name):
        return source_dir / name
    extract(src('phase_three___config.py'), env, assignment_names=[
        'FEATURE_BINS', 'FEATURE_ORDER', 'ACTION_SPACE', 'ACTION_TO_IDX', 'NUM_ACTIONS'])
    env['cfg'] = SimpleNamespace(**{k: env[k] for k in [
        'FEATURE_BINS', 'FEATURE_ORDER', 'ACTION_SPACE', 'ACTION_TO_IDX', 'NUM_ACTIONS']})
    extract(src('phase_three__formalization__core__feature_discretizer.py'), env,
            function_names=['enumerate_state_space'])
    extract(src('phase_three__formalization__core__kernel_builder.py'), env,
            assignment_names=['ACTION_KEYWORDS'], function_names=[
                'classify_step_text', '_step_action_dist', '_state_matches_condition',
                '_state_applicability', 'build_kernel'])
    extract(src('phase_three__formalization__metrics__metric_suite.py'), env,
            function_names=['_safe_row', 'frobenius', 'spectral', 'fisher_rao',
                            'kl_divergence', 'kl_asymmetry', 'all_metrics'])
    extract(src('phase_three__formalization__isomorphism__detector.py'), env,
            function_names=['_classify', '_calibrate_thresholds'])
    cache = json.loads(src('phase_three__analysis__action_classifier_cache.json').read_text())
    cache_hits = []

    class CacheOnly:
        def classify(self, strategy_name, step_text, on_diff=''):
            key = f'{strategy_name}||{step_text[:100]}||{on_diff[:60]}'
            if key not in cache:
                raise RuntimeError('Missing classifier cache; no model/fallback allowed: '+key)
            arr = np.asarray(cache[key], dtype=np.float64)
            if len(arr) != env['cfg'].NUM_ACTIONS or not np.all(np.isfinite(arr)) or arr.sum() <= 0:
                raise RuntimeError('Invalid classifier cache: '+key)
            cache_hits.append(key)
            return arr / arr.sum()

    # Reuse the exact source aggregation method; classify itself is cache-only.
    classifier_source = src('phase_three__formalization__core__llm_action_classifier.py')
    parsed = ast.parse(classifier_source.read_text())
    node = next(n for c in parsed.body if isinstance(c, ast.ClassDef)
                and c.name == 'LLMActionClassifier' for n in c.body
                if isinstance(n, ast.FunctionDef) and n.name == 'strategy_step_distribution')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(classifier_source), 'exec'), env)
    CacheOnly.strategy_step_distribution = env['strategy_step_distribution']
    classifier = CacheOnly()
    strategies = {}
    for f in sorted(source_dir.glob('phase_zero__kb__strategies__S*.json')):
        if f.name.endswith('.meta.json'):
            continue
        s = json.loads(f.read_text())
        strategies[s['id']] = s
    report_ids = {r[k] for r in rows for k in ('strategy_a', 'strategy_b')}
    if set(strategies) != report_ids:
        raise RuntimeError('Strategy input IDs do not exactly match report IDs')
    kernels, structures = {}, {}
    for sid, strategy in sorted(strategies.items()):
        K = env['build_kernel'](strategy, llm_classifier=classifier)
        kernels[sid] = K
        steps = strategy.get('operational_steps', [])
        # Inspect source topology only; never call networkx GED here.
        import re
        recursive = [(step.get('step'), re.search(r'S\d{2}', step.get('on_difficulty') or ''))
                     for step in steps]
        recursive = [(step, hit.group(0)) for step, hit in recursive if hit]
        structures[sid] = {
            'name': strategy.get('name'), 'steps': len(steps),
            'keyword_action_categories': [env['classify_step_text'](s.get('action') or '') for s in steps],
            'recursive_edges_to_shared_external_node': recursive,
            'kernel_shape': list(K.shape), 'numeric_rank_1e_10': int(np.linalg.matrix_rank(K, tol=1e-10)),
            'unique_rows_12_decimals': int(np.unique(np.round(K, 12), axis=0).shape[0]),
        }
    raw = [(a, b, env['all_metrics'](kernels[a], kernels[b])) for a, b in combinations(sorted(kernels), 2)]
    fthr, sthr = env['_calibrate_thresholds']([m for _, _, m in raw])
    saved = {pair(r): r for r in rows}
    mismatches, label_mismatches, confidence_mismatches = [], [], []
    max_abs = {key: 0.0 for key in raw[0][2]}
    exact_labels = {}
    for a, b, metrics in raw:
        old = saved[a, b]
        label, conf = env['_classify'](metrics, fthr, sthr)
        exact_labels[a, b] = label
        if label != old['relation']:
            label_mismatches.append({'pair': [a, b], 'saved': old['relation'], 'rebuilt': label})
        if round(conf, 3) != old['confidence']:
            confidence_mismatches.append({'pair': [a, b], 'saved': old['confidence'], 'rebuilt': round(conf, 3)})
        for key, value in metrics.items():
            delta = abs(value-old[key]); max_abs[key] = max(max_abs[key], delta)
            if round(value, 4) != old[key]:
                mismatches.append({'pair': [a, b], 'metric': key, 'saved': old[key],
                                   'rebuilt_full': value, 'rebuilt_4dp': round(value, 4)})
    return {
        'status': 'completed_cache_only', 'model_calls': 0, 'classifier_fallbacks': 0,
        'cache_entry_count': len(cache), 'cache_lookups': len(cache_hits),
        'unique_cache_entries_used': len(set(cache_hits)),
        'strategies_n': len(strategies), 'pairs_n': len(raw),
        'thresholds_full_precision': {'fisher_thr': fthr, 'spec_thr': sthr},
        'thresholds_3dp': {'fisher_thr': round(fthr, 3), 'spec_thr': round(sthr, 3)},
        'metric_values_compared': len(raw)*len(raw[0][2]),
        'metric_4dp_mismatch_count': len(mismatches), 'metric_4dp_mismatches': mismatches,
        'max_abs_full_precision_vs_report': max_abs,
        'base_label_mismatches': label_mismatches, 'confidence_3dp_mismatches': confidence_mismatches,
        'kernel_rank_counts': dict(Counter(str(s['numeric_rank_1e_10']) for s in structures.values())),
        'structures': structures,
        'claim_boundary': 'Agreement at serialized numeric precision is not proof of original run provenance. SEG/Blackwell values remain supplied-report observations; neither was rerun.',
    }


def cache_coverage(source_dir):
    cache = json.loads((source_dir/'phase_three__analysis__action_classifier_cache.json').read_text())
    coverage = []
    for path in sorted(source_dir.glob('phase_zero__kb__strategies__S*.json')):
        if path.name.endswith('.meta.json'):
            continue
        strategy = json.loads(path.read_text())
        name = strategy.get('name', {}).get('zh', strategy['id'])
        keys = []
        for step in strategy.get('operational_steps', []):
            text = step.get('action', '') or ''
            on_diff = step.get('on_difficulty') or ''
            keys.append(f'{name}||{text[:100]}||{on_diff[:60]}')
            if on_diff:
                keys.append(f'{name}||{on_diff[:100]}||')
        missing = [key for key in keys if key not in cache]
        coverage.append({'id':strategy['id'], 'requested_count':len(keys),
                         'cache_hit_count':len(keys)-len(missing), 'missing_keys':missing})
    return {'entries':len(cache), 'strategies':coverage,
            'requested_count':sum(s['requested_count'] for s in coverage),
            'cache_hit_count':sum(s['cache_hit_count'] for s in coverage),
            'complete_strategies':sum(not s['missing_keys'] for s in coverage),
            'source':'Static key inventory; no model, fallback, GED or LP calls.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--sources', type=Path, help='Optional archived source/cache directory; enables cache-only rebuild')
    ap.add_argument('--output', type=Path, default=Path('recomputed_formalization.json'))
    args = ap.parse_args()
    data_bytes = args.report.read_bytes()
    doc = json.loads(data_bytes)
    rows = doc['reports']
    expected = {tuple(sorted(p)) for p in doc['expected_pairs']}
    all_pairs = [pair(r) for r in rows]
    strategy_ids = sorted({sid for p in all_pairs for sid in p})
    fthr, sthr = calibrate(rows)
    count = lambda key: dict(Counter(r[key] for r in rows))
    detected = {pair(r) for r in rows if accepted(r['combined_relation'])}
    kernel_detected = {pair(r) for r in rows if accepted(r['relation'])}
    hits = expected & detected
    combined_mismatches = []
    rounded_threshold_mismatches, recalibrated_mismatches = [], []
    expected_details, per_pair = [], []
    top30 = {pair(r) for r in sorted(rows, key=lambda r:r['fisher'])[:30]}
    for idx, r in enumerate(rows):
        computed_combined = combine(r['relation'], r['seg_distance'], r['blackwell'])
        if computed_combined != r['combined_relation']:
            combined_mismatches.append({'index': idx, 'pair': pair(r), 'saved': r['combined_relation'], 'computed': computed_combined})
        approx, _ = classify(r, fthr, sthr)
        rounded, _ = classify(r, doc['thresholds']['fisher_thr'], doc['thresholds']['spec_thr'])
        for result, collection in [(approx, recalibrated_mismatches), (rounded, rounded_threshold_mismatches)]:
            if result != r['relation']:
                collection.append({'index': idx, 'pair': pair(r), 'saved':r['relation'], 'recomputed':result})
        detail = {
            'index': idx, 'json_path': f'reports[{idx}]', 'pair': list(pair(r)),
            'fisher': r['fisher'], 'spectral': r['spectral'], 'kl_log_asym': r['kl_log_asym'],
            'seg_distance': r['seg_distance'], 'blackwell': r['blackwell'],
            'relation': r['relation'], 'combined_relation': r['combined_relation'],
            'approx_gates_using_recalibrated_threshold': {
                'fisher_tight': r['fisher'] < .7*fthr,
                'fisher_loose': r['fisher'] < fthr,
                'spectral_ok': r['spectral'] < sthr,
                'kl_symmetric': abs(r['kl_log_asym']) < .3,
                'seg_downgrade_triggered': r['relation']=='iso' and r['seg_distance'] > .8,
                'seg_only_rescue_triggered': r['relation']=='distinct' and r['seg_distance'] < .3,
                'blackwell_upgrade_available': r['blackwell']=='both_dominate',
            },
            'kernel_accepted': accepted(r['relation']), 'final_accepted': accepted(r['combined_relation']),
            'fisher_rank_min': 1+sum(v['fisher'] < r['fisher'] for v in rows),
            'seg_rank_min': 1+sum(v['seg_distance'] < r['seg_distance'] for v in rows),
            'blackwell_checked_matches_top30': (r['blackwell']!='not_checked') == (pair(r) in top30),
        }
        per_pair.append(detail)
        if pair(r) in expected:
            expected_details.append(detail)
    def result_for(selected):
        return {'accepted_pairs':len(selected), 'expected_hits':[list(p)for p in sorted(expected & selected)],
                'expected_recall':len(expected & selected)/len(expected)}
    output = {
        'input': {'path':str(args.report), 'sha256':hashlib.sha256(data_bytes).hexdigest(),
                  'bytes':len(data_bytes), 'provenance':'User-supplied report; no embedded commit or run manifest'},
        'scope':'Recount and logic replay of saved report; optional cached kernel rebuild; no models, network, full GED, or LP rerun',
        'shape': {'strategies':strategy_ids, 'strategies_n':len(strategy_ids), 'pairs_n':len(rows),
                  'n_choose_2':math.comb(len(strategy_ids),2), 'unique_pairs_n':len(set(all_pairs)),
                  'missing_pairs':[list(p)for p in sorted(set(combinations(strategy_ids,2))-set(all_pairs))]},
        'thresholds': {'saved':doc['thresholds'], 'recalculated_from_4dp_reports':{'fisher_thr':fthr,'spec_thr':sthr},
            'recalculated_rounded_to_3dp_matches_saved':round(fthr,3)==doc['thresholds']['fisher_thr'] and round(sthr,3)==doc['thresholds']['spec_thr'],
            'fisher_tight_approx':.7*fthr,
            'class_mismatches_using_rounded_saved_thresholds':rounded_threshold_mismatches,
            'class_mismatches_using_recalibrated_4dp_metrics':recalibrated_mismatches,
            'rounding_boundary_note':'Full-precision decisions cannot in general be recovered from rounded report fields.'},
        'distributions':{key:describe([r[key] for r in rows]) for key in ['fisher','spectral','seg_distance','kl_log_asym']},
        'counts':{key:count(key)for key in ['relation','combined_relation','blackwell']},
        'transition_counts':dict(Counter(r['relation']+' -> '+r['combined_relation']for r in rows)),
        'validation':{'expected_pairs':[list(p)for p in sorted(expected)], 'recomputed_hits':[list(p)for p in sorted(hits)],
                      'recomputed_missed':[list(p)for p in sorted(expected-hits)], 'recomputed_recall':len(hits)/len(expected),
                      'matches_report_hits':sorted(map(tuple,doc['hits']))==sorted(hits),
                      'matches_report_missed':sorted(map(tuple,doc['missed']))==sorted(expected-hits),
                      'matches_report_recall':doc['recall']==len(hits)/len(expected)},
        'rules':{'combined_label_mismatches':combined_mismatches,
                 'blackwell_checks_exactly_fisher_top30':all(d['blackwell_checked_matches_top30']for d in per_pair),
                 'accepted_with_blackwell_neither':sum(accepted(r['combined_relation']) and r['blackwell']=='neither' for r in rows),
                 'accepted_with_blackwell_unchecked':sum(accepted(r['combined_relation']) and r['blackwell']=='not_checked' for r in rows),
                 'weak_iso_above_seg_0_8':sum(r['relation']=='weak_iso' and r['seg_distance']>.8 for r in rows),
                 'explicit_law_invariant_gate_present':False,
                 'published_vs_comment_note':'Source comment says kernel AND SEG<0.6 AND Blackwell-equivalent; actual ordered elif rules differ.'},
        'sensitivity_only_not_new_performance':{
            'kernel_labels_before_SEG':result_for(kernel_detected),
            'actual_final':result_for(detected),
            'same_SEG_0_8_filter_on_all_kernel_candidates':result_for({pair(r)for r in rows if accepted(r['relation']) and r['seg_distance']<=.8}),
            'comment_kernel_iso_AND_SEG_under_0_6_AND_Blackwell_both':result_for({pair(r)for r in rows if r['relation']=='iso' and r['seg_distance']<.6 and r['blackwell']=='both_dominate'}),
        },
        'expected_pair_details':expected_details,
        'all_pair_gate_audit':per_pair,
    }
    if args.sources:
        output['static_cache_coverage'] = cache_coverage(args.sources)
        try:
            output['cached_kernel_rebuild'] = rebuild_kernels(args.sources, rows)
        except Exception as exc:
            output['cached_kernel_rebuild'] = {'status':'stopped','error':str(exc),'model_calls':0,
                                               'note':'No classifier fallback or model call was attempted.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    brief = {'output':str(args.output), 'pairs':len(rows), 'recall':output['validation']['recomputed_recall'],
             'counts':output['counts'], 'rebuild':{k:v for k,v in output.get('cached_kernel_rebuild',{}).items()
                if k in ['status','error','cache_lookups','unique_cache_entries_used','metric_4dp_mismatch_count','base_label_mismatches','confidence_3dp_mismatches','thresholds_3dp','kernel_rank_counts']}}
    print(json.dumps(brief,ensure_ascii=False))


if __name__ == '__main__':
    main()
