"""Prepare the fixed 0.160/0.161 follow-up research experiment."""
import copy
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
OLD = ROOT / 'artifacts/confirmation_investment_20261002/instrument_seen'
original = json.loads((OLD / 'protocol.json').read_text())
old_runtime = 'artifacts/confirmation_investment_20261002/runtime/src'
new_runtime = 'artifacts/marginal_research_investment_20261002/runtime/src'
cohorts = ('seen_regression', 'fresh_successors')
for cohort in cohorts:
    protocol = copy.deepcopy(original)
    protocol.update(title='Marginal follow-up research value and subsequent task investment',
        baseline_version='0.160.0', candidate_version='0.161.0', cohort=cohort,
        baseline_runtime_reference=old_runtime,
        candidate_variant='Use the marginal value of additional condition research and transferable follow-up experience.',
        primary_endpoint='All-task independent response MSE, solved<=1e-6, false completion, charged observation/measurement/compute and actual process CPU; additionally later-round outcomes and ablation of new follow-up experience.',
        hypothesis_test='The marginal return of additional condition research should guide current expenditure and improve subsequent-task decisions within identical budgets.',
        evaluation='Both cohorts and all arms finish learning before any new terminal evaluation. Twelve seen 0.160 chains and their terminal evaluations are reused explicitly. No result-based exclusions or tuning.',
        source_accounting='All new arms begin with an empty library. The deletion arm removes only action_experience contexts containing :condition_followup_research: before each task/revisit; all other knowledge and current-task paid evidence remain.',
        learning_barrier_outputs=[str((BASE/name).relative_to(ROOT)) for name in cohorts],
        candidate_source_status='Awaiting root GO; baseline source already frozen.',
        execution={'maximum_workers':16,'status':'baseline learning only; candidates await GO'},
        prior_protocol=str((OLD/'protocol.json').relative_to(ROOT)),
        scope='seen regression' if cohort == 'seen_regression' else 'new same-factory apparatus with repeated mechanisms')
    protocol['arms'] = {
        'old_cumulative': {'runtime':old_runtime, 'knowledge':'empty', 'result_source':'reused 0.160 instrument_seen new_cumulative' if cohort == 'seen_regression' else 'new run'},
        'new_cumulative': {'runtime':new_runtime, 'knowledge':'empty', 'result_source':'new run'}}
    if cohort == 'fresh_successors':
        protocol['arms']['no_marginal_history'] = {'runtime':new_runtime, 'knowledge':'empty', 'result_source':'new run',
            'remove_action_experience_context':':condition_followup_research:'}
        sequence = [0,1,2,3,4,5,6,2,1,4,3,5,0,6]
        protocol['stage_sequence'] = sequence
        protocol['stages'] = [f'round{1 + position//7}_{original["stages"][stage]}' for position,stage in enumerate(sequence)]
        protocol['physical_stage_semantics'] = [original['physical_stage_semantics'][stage] for stage in sequence]
        protocol['roots'] = {f'root{index:02d}':{'tasks':[{'apparatus_seed':seed,'seed':seed+610003+1009*position,'stage':stage}
            for position,stage in enumerate(sequence)]} for index,seed in enumerate(range(9935101,9935113),1)}
        for key in ('observation','measurement','compute'):
            protocol['limits']['chain_'+key] *= 2
    directory = BASE/cohort
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    if cohort == 'seen_regression':
        for root in protocol['roots']:
            target = directory/'runs'/'old_cumulative'/root
            target.mkdir(parents=True,exist_ok=True)
            for name in ('checkpoint.json.gz','evaluation.json'):
                path = target/name
                if not path.exists():
                    path.symlink_to(OLD/'runs'/'new_cumulative'/root/name)
print({'new_learning':48,'new_terminal_task_arm_scores':588,'reused_learning':12,'compared_terminal_task_arm_scores':672})
