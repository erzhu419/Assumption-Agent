"""Read-only independent recomputation of committed frontier receipts; no model calls."""
from pathlib import Path
import os
import collections, hashlib, json, math, sys

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get('HEGEL_M6_REPO', str(Path(__file__).resolve().parent.parent / 'repo_m6_b1d0c3e99')))
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / 'Hegel Machine/src'))
from hegel_machine.cognition.m5_live_gpt_and_provenance import verify_decision_provenance_chain

ROOT = REPO / 'Hegel Machine/docs/evidence'
BUNDLES = ['m6_frontier_scout_80ep_20261121', 'm6_1_frontier_28turn_40ep_20261121', 'm6_2_frontier_28turn_40ep_20261121']
all_rows, checks, aggregates, traces = [], [], [], []
for bundle in BUNDLES:
    root = ROOT / bundle
    mismatches = []
    sha_lines = (root / 'SHA256SUMS.txt').read_text().splitlines()
    for line in sha_lines:
        digest, rel = line.split('  ', 1)
        file = root / rel
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            mismatches.append(rel)
    rows = json.loads((root / 'all_episode_summaries.json').read_text())
    chains_bad, summary_bad, verification_bad, tokens_bad = [], [], [], []
    for row in rows:
        ep = root / 'episodes' / f"{row['wave']}__{row['model']}__{row['arm']}__{row['task_id']}"
        saved = json.loads((ep / '00_episode_summary.json').read_text())
        if saved != row: summary_bad.append(ep.name)
        dag = json.loads((ep / '01_decision_provenance_dag.json').read_text())
        records = dag['records']
        check = verify_decision_provenance_chain(records)
        if not check['valid']: chains_bad.append(ep.name)
        verifier = json.loads((ep / '03_official_verifier_report.json').read_text())
        if bool(verifier['strict_pass']) != bool(row['strict_pass']): verification_bad.append(ep.name)
        for key in ['prompt_tokens', 'completion_tokens', 'cached_tokens']:
            got = sum(r['cost_ledger_delta'].get(key + '_delta', 0) for r in records)
            if got != row[key]: tokens_bad.append([ep.name, key, got, row[key]])
        ctrf = verifier.get('ctrf_summary', {}).get('summary', {})
        rr = dict(row, bundle=bundle, episode_dir=str(ep), ctrf_passed=ctrf.get('passed', 0), ctrf_total=ctrf.get('tests', 0),
                  dag_sha256=hashlib.sha256((ep / '01_decision_provenance_dag.json').read_bytes()).hexdigest(),
                  source_run_id=records[0]['run_id'], failed_tests=verifier.get('ctrf_summary', {}).get('failed_tests', []))
        all_rows.append(rr)
        traces.append({'bundle':bundle, 'episode':ep.name, 'strict_pass':row['strict_pass'],
                       'records':[{'type':r['decision_type'], 'reason':r['rationale_summary'], 'chosen':r['chosen_option'],
                                   'stdout':r.get('extra_artifacts', {}).get('stdout_tail'),
                                   'stderr':r.get('extra_artifacts', {}).get('stderr_tail'),
                                   'nogood':r.get('extra_artifacts', {}).get('nogood_update')}
                                  for r in records]})
    checks.append({'bundle':bundle, 'sha_files':len(sha_lines), 'episodes':len(rows), 'sha_mismatches':mismatches,
                   'bad_chains':chains_bad, 'bad_summaries':summary_bad, 'bad_verifications':verification_bad, 'bad_token_totals':tokens_bad})
    for model, arm in sorted(set((r['model'],r['arm']) for r in rows)):
        subset=[r for r in all_rows if (r['bundle'],r['model'],r['arm'])==(bundle,model,arm)]
        ag={'bundle':bundle,'model':model,'arm':arm,'n':len(subset),'passes':sum(r['strict_pass'] for r in subset),
            'wave1_passes':sum(r['strict_pass'] for r in subset if r['wave']=='wave_1'),
            'wave2_passes':sum(r['strict_pass'] for r in subset if r['wave']=='wave_2'),
            'ctrf_passed':sum(r['ctrf_passed'] for r in subset),'ctrf_total':sum(r['ctrf_total'] for r in subset),
            'turns':sum(r['turns_used'] for r in subset), 'total_tokens':sum(r['total_tokens'] for r in subset),
            'effective_billed_tokens':sum(r['effective_billed_tokens'] for r in subset),
            'episode_wall_seconds':sum(r['wall_ms'] for r in subset)/1000,
            'nogoods':sum(r['nogood_count'] for r in subset)}
        aggregates.append(ag)

groups=collections.defaultdict(list)
for r in all_rows:groups[r['dag_sha256']].append({'bundle':r['bundle'],'episode':Path(r['episode_dir']).name})
duplicates=[v for v in groups.values() if len(v)>1]
sol=[r for r in all_rows if r['model']=='gpt-6.2-sol']
matrix=[]
for task in sorted(set(r['task_id'] for r in sol)):
    record={'task_id':task}
    for r in sol:
        if r['task_id']!=task:continue
        key=('14T_Bare' if r['arm']=='Bare_ReAct_LiveGPT' else 'M6.0_14T') if r['bundle']==BUNDLES[0] else ('28T_Bare' if r['arm']=='Bare_ReAct_LiveGPT' else ('M6.1_28T' if r['bundle']==BUNDLES[1] else 'M6.2_28T'))
        record['wave']=r['wave']
        record[key]={'pass':r['strict_pass'],'turns':r['turns_used'],'subtests':f"{r['ctrf_passed']}/{r['ctrf_total']}",
                     'tokens':r['total_tokens'],'failure':r['failure_taxonomy'],'failed_tests':r['failed_tests']}
    matrix.append(record)
pair={'hegel_only':[], 'bare_only':[], 'both_pass':[], 'both_fail':[]}
for r in matrix:
    h,b=r['M6.2_28T']['pass'],r['28T_Bare']['pass']
    key='both_pass' if h and b else ('both_fail' if not h and not b else ('hegel_only' if h else 'bare_only'))
    pair[key].append(r['task_id'])
n=len(pair['hegel_only'])+len(pair['bare_only']);k=min(len(pair['hegel_only']),len(pair['bare_only']))
pair['mcnemar_exact_two_sided_p']=min(1,2*sum(math.comb(n,i) for i in range(k+1))/2**n) if n else 1
result={'committed_occurrences':len(all_rows),'unique_trajectory_hashes':len(groups),'checks':checks,'aggregates':aggregates,
        'duplicate_trajectory_groups':duplicates,'matrix':matrix,'M6_2_vs_Bare_paired':pair}
(OUT/'frontier_recomputed.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
(OUT/'frontier_episode_index.json').write_text(json.dumps(all_rows,indent=2,ensure_ascii=False))
(OUT/'frontier_trace_digest.json').write_text(json.dumps(traces,indent=2,ensure_ascii=False))
print(json.dumps({k:result[k] for k in ['committed_occurrences','unique_trajectory_hashes','checks','aggregates','M6_2_vs_Bare_paired']},indent=2,ensure_ascii=False))
