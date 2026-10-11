"""Prepare three fixed arms without freezing source or running research."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PRIOR = ROOT / 'artifacts/pair_confirmation_investment_20261003/successor_regression'
ANCHOR = ROOT / 'artifacts/shared_condition_research_20261003/successor_regression'
output = BASE / 'successor_regression'
protocol = deepcopy(json.loads((PRIOR / 'protocol.json').read_text()))
protocol.update(
    title='Concrete pair confirmations compete for shared research resources',
    baseline_version='0.173.0', candidate_version='0.174.0', quality_anchor_version='0.164.0',
    hypothesis_test='Queue a specific unresolved frozen pair in the shared research pool and evaluate the downstream contribution of complete independent-pair visit cost experience local to its specific pending question. This is not general cross-task method knowledge.',
    primary_endpoint='All-task and last-seven-stage MSE, solved<=1e-6, false completion, all observation/measurement/compute and CPU, task/root wins-ties-losses, every failure/recovery, queued versus actually executed pair confirmations, paid discrimination and complete cost feedback.',
    evaluation='All36 new long learning chains finish before504 new terminal scores. Reuse24 old chains/336 scores from actual0.173 and0.164. Preserve every root/task/seed/limit; no terminal-driven source changes or repeat learning.',
    baseline_runtime_reference='artifacts/pair_confirmation_investment_20261003/runtime/src',
    candidate_variant='Natural session queue and question-local cost-recording interfaces. All arms retain0.173 within-visit pair procurement, coherent forecasting, actual resource payment and ordinary research.',
    development_selection='Use existing small paid prefixes or a small new fixture supplied by the implementation tests. Do not scan or copy another large checkpoint. Exact shared-session development scope remains pending.',
    prior_protocol=str((PRIOR / 'protocol.json').relative_to(ROOT)),
    candidate_source_status='Prepared only; awaiting stable source, tests and root GO.',
    candidate_freeze='Only after final production tests and explicit root GO.',
    execution={'maximum_workers':12,'status':'prepared_not_started'},
    learning_barrier_outputs=[str(output.relative_to(ROOT))],
    source_accounting='All new arms start empty. prior_pair_queue disables only the new pair opportunity queue. no_pair_cost_feedback keeps the queue, actual payments, observations, programs and evidence feedback but suppresses only cost-experience writes from the new independent pair visits. No production flags.',
    development_causal_correction='0.173 improved root12 p9 indirectly through the shared path following position3 evidence; it did not directly confirm position9. The new study distinguishes access to globally competing pair work from its downstream cost-history contribution.',
    attribution_scope='Full versus no_pair_cost_feedback isolates this cost-history path. Full versus prior_pair_queue combines the new queue and its downstream evidence/cost effects; it does not separately identify every evidence mechanism.',
)
runtime = str((BASE / 'runtime/src').relative_to(ROOT))
protocol['arms'] = {
    'anchor_164': {'runtime':'artifacts/shared_condition_research_20261003/runtime/src','knowledge':'empty','result_source':'reused0.164 long cohort'},
    'old_cumulative': {'runtime':'artifacts/pair_confirmation_investment_20261003/runtime/src','knowledge':'empty','result_source':'reused0.173 long cohort'},
    'new_cumulative': {'runtime':runtime,'knowledge':'empty','result_source':'new run'},
    'prior_pair_queue': {'runtime':runtime,'knowledge':'empty','result_source':'new run',
        'ablation_interface':'Study monkeypatch UnifiedResearchSession._offer_pair_confirmations(self,pending) to return pending. Preserve0.173 within-visit pair procurement.'},
    'no_pair_cost_feedback': {'runtime':runtime,'knowledge':'empty','result_source':'new run',
        'ablation_interface':'Study monkeypatch pair_confirmation_revisits.record_pair_revisit_cost(question,visit_costs) to return an unchanged deep copy. Skip only question-local execution_costs samples from new independent pair visits; actual ledger, evidence, information history, programs and ordinary cost history stay.'},
}
output.mkdir(parents=True,exist_ok=True)
(output / 'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
for arm, source in (('anchor_164',ANCHOR),('old_cumulative',PRIOR)):
    for root in protocol['roots']:
        target = output / 'runs' / arm / root
        target.mkdir(parents=True,exist_ok=True)
        for name in ('checkpoint.json.gz','evaluation.json'):
            path = target / name
            if not path.exists():
                path.symlink_to(source / 'runs/new_cumulative' / root / name)
development = {
    'status':'pending_small_development_scope_and_GO',
    'existing_paid_fixture':'artifacts/coherent_confirmation_20261003/paid_confirmation_cases.json.gz',
    'existing_paid_outcomes':'artifacts/pair_confirmation_investment_20261003/development_result.json',
    'scope':'Exercise concrete-pair queue construction, real shared-pool selection and isolated cost feedback once their interfaces are settled. Existing small fixture lacks an exact complete historical session, so no invented source balance or hidden final knowledge.',
    'budget':'Set the development allowance before any new outcome; formal per-task320/320/320000 and full14-task pool remain unchanged.',
    'terminal_scores':0,'external_model_calls':0,
}
(BASE / 'development_protocol.json').write_text(json.dumps(development,indent=2)+'\n')
print({'prepared_new_learning':36,'prepared_terminal_scores':504,
       'reused_chains':24,'reused_scores':336,'executed_new_runs':0})
