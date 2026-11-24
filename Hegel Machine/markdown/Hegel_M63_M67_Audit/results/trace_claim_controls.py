"""Check causal narrative claims against sealed same-trajectory records, read-only."""
from pathlib import Path
import hashlib,json,os,re

REPO=Path(os.environ.get('HEGEL_M67_REPO',str(Path(__file__).resolve().parents[2]/'repo_m67_948e2d57a')))
EV=REPO/'Hegel Machine/docs/evidence'
OUT=Path(__file__).resolve().parent
BUNDLE=EV/'m6_7_host_mechanical_wormhole_frontier_40ep_20261122'

def episode(task):
    p=next((BUNDLE/'episodes').glob('*Hegel*__tb__'+task))
    return p,json.loads((p/'00_episode_summary.json').read_text()),json.loads((p/'01_decision_provenance_dag.json').read_text())['records']

def turn_of(record):
    m=re.search(r'D_TURN_(\d+)',record['decision_type']);return int(m.group(1)) if m else None

p,s,records=episode('clinical-metadata-recovery')
gate=next(i for i,r in enumerate(records) if r['decision_type']=='D_HOST_MECHANICAL_WORMHOLE_GATE_AUDIT')
pre_gate_artifact_change=[]
post_gate_changes=[]
for i,r in enumerate(records):
    for kind,items in r.get('extra_artifacts',{}).get('filesystem_diff',{}).items():
        for item in items:
            if item.get('path')=='/root/results/recovered_metadata.tsv':
                (pre_gate_artifact_change if i<gate else post_gate_changes).append({'event':r['decision_type'],'kind':kind,**item})
artifact=p/'04_submitted_artifacts_snapshot/recovered_metadata.tsv'
clinical={'last_pregate_change':pre_gate_artifact_change[-1], 'postgate_artifact_changes':post_gate_changes,'final_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'matches_pregate_hash':hashlib.sha256(artifact.read_bytes()).hexdigest().startswith(pre_gate_artifact_change[-1]['sha16']),'official_pass':s['strict_pass'],'final_rationale':[r['rationale_summary'] for r in records if r['decision_type'].endswith('AGENT_DECISION')][-1]}

p,s,records=episode('symbolic-regression')
gate=next(i for i,r in enumerate(records) if r['decision_type']=='D_HOST_MECHANICAL_WORMHOLE_GATE_AUDIT')
symbolic={'preceding_turn':next(turn_of(r) for r in reversed(records[:gate]) if turn_of(r)),'subsequent_decisions':[{'event':r['decision_type'],'rationale':r['rationale_summary']}for r in records[gate:] if r['decision_type'].endswith('AGENT_DECISION')],'gate_receipt':records[gate]['chosen_option'],'final_failure':json.loads((p/'03_official_verifier_report.json').read_text())['ctrf_summary']['failed_tests']}

p,s,records=episode('certified-sparse-regression')
gate=next(i for i,r in enumerate(records) if r['decision_type']=='D_HOST_MECHANICAL_WORMHOLE_GATE_AUDIT')
certificate={'preceding_turn':next(turn_of(r)for r in reversed(records[:gate])if turn_of(r)),'preceding_decision':records[gate-1]['decision_type'],'gate_receipt':records[gate]['chosen_option'],'max_turns':s['max_turns'],'final_turns':s['turns_used']}

p,s,records=episode('sparse-network-assimilation')
durations={k:sum(r.get('cost_ledger_delta',{}).get('wall_ms_delta',0)for r in records if (r['decision_type'].endswith('AGENT_DECISION') if k=='model' else r['decision_type'].endswith('CONTAINER_EXECUTION') if k=='container_commands' else not(r['decision_type'].endswith('AGENT_DECISION')or r['decision_type'].endswith('CONTAINER_EXECUTION'))))/1000 for k in ['model','container_commands','other_ledger']}
sparse={'episode_wall_seconds':s['wall_ms']/1000,'ledger_seconds':durations,'unattributed_wall_seconds':s['wall_ms']/1000-sum(durations.values()),'reported_calls':s.get('skill_activation_count'),'telemetry_has_duration':any('elapsed' in key or 'duration' in key for a in s.get('skill_activations',[])for key in a),'failed_tests':json.loads((p/'03_official_verifier_report.json').read_text())['ctrf_summary']['failed_tests']}

result={'clinical_gate_did_not_change_passing_artifact':clinical,'symbolic_gate_timing':symbolic,'certificate_gate_timing':certificate,'sparse_network_time_attribution':sparse}
(OUT/'trace_claim_controls_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
