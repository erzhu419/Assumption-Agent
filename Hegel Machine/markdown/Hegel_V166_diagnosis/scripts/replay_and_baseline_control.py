#!/usr/bin/env python3
"""Offline fixed-source replay plus a narrow non-preregistered baseline control.

No LLM/network calls, no source changes. The only runner shim supplies the known
commit for its metadata-only `git rev-parse HEAD` call because a downloaded
source snapshot need not contain .git. All model/simulator/solver code is intact.
The extra control enables an already implemented optional prior argument using
ordinary's own source archive and caches that decision, at a billed proxy cost.
It is a diagnostic policy intervention, not a new preregistered benchmark result.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import mean
import sys
import types
from unittest.mock import patch

COMMIT = '675f52612a9d0f1fb208da8bdb0552de3c571758'
RUNNER_GIT_BLOB = 'c6f1cde5a1391315abe13cedb4f32c226d8b91ff'
sys.dont_write_bytecode = True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-root', required=True, type=Path)
    ap.add_argument('--output-dir', required=True, type=Path)
    args = ap.parse_args()
    repo = args.repo_root.resolve()
    out = args.output_dir.resolve()
    if out == repo or repo in out.parents:
        ap.error('Use an output directory outside the source snapshot.')
    out.mkdir(parents=True, exist_ok=True)
    evidence = repo/'docs/evidence/v16_6_cross_domain_wormhole_20261119'
    manifest = json.loads((evidence/'experiment_manifest.json').read_text())
    hashes = {rel:hashlib.sha256((repo/rel).read_bytes()).hexdigest()
              for rel in manifest['module_sha256']}
    if hashes != manifest['module_sha256']:
        ap.error('Module bytes do not match the saved fixed-study manifest.')
    runner_file = repo/'scripts/run_v16_6_m1_cross_domain_wormhole_study.py'
    raw = runner_file.read_bytes()
    blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if blob != RUNNER_GIT_BLOB:
        ap.error('Runner bytes do not match the audited fixed commit.')

    # Namespace package shims avoid executing an unrelated top-level __init__.
    for name, directory in [('hegel_machine',repo/'src/hegel_machine'),
                            ('hegel_machine.cognition',repo/'src/hegel_machine/cognition')]:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(directory)]
        sys.modules[name] = pkg
    spec = importlib.util.spec_from_file_location('v166_study_runner',runner_file)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    original_check_output = runner.subprocess.check_output
    def metadata_git_only(cmd, *a, **kw):
        if cmd == ['git','rev-parse','HEAD']:
            return COMMIT+'\n'
        return original_check_output(cmd,*a,**kw)
    dev_sources=[]
    dev_rollouts=[]
    gate_snapshot={}
    original_formation=runner.HegelV166Researcher.form_and_revise_from_experience
    original_calibration=runner._calibrate_gate_on_dev_clusters
    def capture_formation(self,cluster_id,*a,**kw):
        logs=original_formation(self,cluster_id,*a,**kw)
        if cluster_id.startswith('dev_cluster_'):
            dev_sources.extend(deepcopy(logs))
        return logs
    def capture_calibration():
        gate,records=original_calibration()
        dev_rollouts.extend(deepcopy(records))
        gate_snapshot.update({'paired_dev_records':deepcopy(gate.paired_dev_records),
                              'exploration_uses':deepcopy(gate.exploration_uses_by_signature),
                              'margin':gate.margin,'probe_budget_fraction_cap':gate.probe_budget_fraction_cap})
        return gate,records
    with patch.object(runner.subprocess,'check_output',metadata_git_only), \
         patch.object(runner.HegelV166Researcher,'form_and_revise_from_experience',capture_formation), \
         patch.object(runner,'_calibrate_gate_on_dev_clusters',capture_calibration):
        replay = runner.run_study(out/'replayed_evidence')
    original = json.loads((evidence/'paired_results.json').read_text())
    replay_comparison = {
        'exact_paired_results_equal': replay == original,
        'expected_commit':COMMIT,
        'runner_git_blob':blob,
        'module_sha256':hashes,
        'metadata_only_shim':'git rev-parse HEAD returns supplied fixed commit; no scientific computation patched.',
        'output_source_was_not_modified':True,
        'rerun_scope':'All 4 eval clusters, 7 arms and development calibration; deterministic local Python simulator, no LLM.',
    }
    (out/'replay_verification.json').write_text(json.dumps(replay_comparison,indent=2)+'\n')
    dev_account={
        'scope':'Passive instrumentation of unchanged development calibration; these episode costs are absent from published eval full-lifecycle totals.',
        'source_episode_count':len(dev_sources),
        'paired_target_rollout_count':len(dev_rollouts),
        'total_episode_executions':len(dev_sources)+2*len(dev_rollouts),
        'source_composite_cost':sum(r['costs']['composite_cost'] for r in dev_sources),
        'target_pair_composite_cost':sum(r['hegel_composite_cost']+r['direct_composite_cost'] for r in dev_rollouts),
        'gate_snapshot':gate_snapshot,'source_episodes':dev_sources,'paired_rollouts':dev_rollouts,
    }
    dev_account['total_composite_cost']=dev_account['source_composite_cost']+dev_account['target_pair_composite_cost']
    (out/'development_costs.json').write_text(json.dumps(dev_account,ensure_ascii=False,indent=2)+'\n')

    from hegel_machine.cognition.cross_domain_arms import StrongOrdinaryResearcher, solve_resource_1d_direct
    from hegel_machine.cognition.cross_domain_clusters import (
        PublicOrderedBoundaryTask,make_cluster_a_suite,make_cluster_b_suite,
        make_cluster_c_same_service_suite)

    class ArchivePriorControl(StrongOrdinaryResearcher):
        """One change: actually consume/archive-cache ordinary's own induced family.

        No environment/task names or hidden truth choose a prior. Mixed histories
        still require the native direct solver's target 3rd-point check and fresh
        confirmation. Reconstruction uses the existing 28-unit tariff, charged
        once per source cluster; cached selection is allowed by the main contract.
        """
        def solve_target(self, name, task):
            if isinstance(task,PublicOrderedBoundaryTask) or 'u' not in task.control_bounds:
                return super().solve_target(name,task)
            task.meter.charge('retrieve',compute=2.0)
            if 'source_prior' not in self.rebuilt_cache:
                if not task.meter.charge('on_demand_reconstruction',compute=28.0):
                    return solve_resource_1d_direct(task)
                families = [r.get('selected_family') for r in self.raw_archive
                            if r.get('observations')]
                self.rebuilt_cache['source_prior'] = (
                    'rational_saturation' if families.count('rational_saturation') >= 2 else None)
            return solve_resource_1d_direct(
                task,reconstructed_prior_family=self.rebuilt_cache['source_prior'])

    control_rows=[]
    summaries=[]
    original_targets=[json.loads(s) for s in (evidence/'target_episodes.jsonl').read_text().splitlines() if s.strip()]
    for cid,seed in runner.EVAL_CLUSTERS:
        ca,cb,cc=make_cluster_a_suite(seed),make_cluster_b_suite(seed),make_cluster_c_same_service_suite(seed)
        ctl=ArchivePriorControl()
        source_logs=ctl.run_source_and_nearmiss_phase(ca,cb)
        rr=[]
        for group,suite in [('Cluster_A',ca),('Cluster_B',cb),('Cluster_C',cc)]:
            for name,(factory,evaluator) in suite['targets']:
                task=factory(); sub=ctl.solve_target(name,task); ev=evaluator(sub)
                record={'cluster_id':cid,'family':group,'target_name':name,
                        'submission':{k:v for k,v in sub.items() if k!='predict_fn'},
                        'evaluation':ev,'costs':task.meter.snapshot()}
                control_rows.append(record); rr.append(record)
        sc=sum(r['costs']['composite_cost'] for r in source_logs)
        metrics=('observation_cost','measurement_cost','compute_cost','composite_cost')
        tgt={k:sum(r['costs'][k] for r in rr) for k in metrics}
        comps={}
        for arm in ['Hegel_V16_6','O_strong']:
            old=[r for r in original_targets if r['cluster_id']==cid and r['arm']==arm]
            comps['control_minus_'+arm]={k:tgt[k]-sum(r['costs'][k] for r in old) for k in metrics}
        summaries.append({'cluster_id':cid,'source_composite':sc,'target':tgt,
                          'end_to_end_composite':sc+tgt['composite_cost'],
                          'strict_pass_count':sum(bool(r['evaluation']['strict_pass']) for r in rr),
                          'false_confirmations':sum(bool(r['evaluation'].get('false_confirmation')) for r in rr),
                          'cached_prior':ctl.rebuilt_cache,'comparisons':comps})
    control={
        'scope':'Non-preregistered diagnostic intervention, not a new fairness-complete benchmark or proof of optimal ordinary performance.',
        'change':'Reads O_strong own computed selected_family values from its paid source archive; caches a rational prior if two source fits support it; calls existing optional prior argument for every 1D target. No target-name routing; fresh native checks retained.',
        'tariff_note':'28 compute proxy units once per cluster for reconstruction, plus existing ordinary per-target retrieval and native fitting tariffs.',
        'per_cluster':summaries,
        'mean_target':{k:mean(s['target'][k] for s in summaries) for k in metrics},
        'mean_end_to_end_composite':mean(s['end_to_end_composite'] for s in summaries),
        'total_strict_pass':sum(s['strict_pass_count'] for s in summaries),
        'total_false_confirmations':sum(s['false_confirmations'] for s in summaries),
        'episodes':control_rows,
    }
    (out/'ordinary_archive_control.json').write_text(json.dumps(control,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'replay':replay_comparison,
                      'control':{k:v for k,v in control.items() if k not in ('episodes','per_cluster')},
                      'control_per_cluster':summaries},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
