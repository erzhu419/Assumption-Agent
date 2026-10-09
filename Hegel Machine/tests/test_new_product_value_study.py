"""The product ablation preserves paid work and scores only completed forks."""

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def runner(name='run_new_product_value_study'):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actual_product_deletion_preserves_old_knowledge_methods_and_paid_reports():
    study = runner()
    plan = study.cohort_protocol()
    original = json.loads((Path(plan['cases'][0]['source_case']) / 'producer_training.json').read_text())['state']
    retained, products = study.intervene(original, 'retain_product')
    removed, _ = study.intervene(original, 'remove_product')
    assert retained == original
    assert products == ['concept_0004']
    assert {key: value for key, value in removed.items() if key != 'knowledge'} == {
        key: value for key, value in original.items() if key != 'knowledge'}
    before, after = original['knowledge'], removed['knowledge']
    assert after['concepts'] == before['concepts'][:-1]
    assert {key for key in before if before[key] != after[key]} == {
        'concepts', 'abstracts', 'deleted_ids', 'meta_priors', 'definitions'}
    assert all(record['source_concept_id'] not in products for record in after['abstracts'])
    assert all(record['definition']['source_concept_id'] not in products for record in after['definitions'])
    assert after['meta_priors'] == before['meta_priors'][:-1]
    assert len(plan['cases']) == 12
    assert [case['seed'] for case in plan['cases']] == [67107001 + 100003*i for i in range(12)]
    assert 'post-hoc' in plan['scope']


def test_real_native_forks_score_after_training_and_include_production(tmp_path, monkeypatch):
    study, paid = runner(), runner('run_paid_revisit_study')
    plan = paid.protocol(67009011)
    plan['research_config'].update(observation_budget=96, compute_budget=12000, max_rounds=1)
    for task in [*plan['return_tasks'], *plan['competition_tasks'], *plan['tasks']]:
        task['task_index'] = 0
    plan['tasks'][1]['task_index'] = 1
    plan['evaluation'] = {'length': 8, 'trajectories': 1, 'seed_offset': 1000003}
    source = tmp_path / 'source'
    paid.prepare_case(plan, source)
    output = tmp_path / 'attribution'
    load = study._load
    scored = []

    def observed_load(path, name):
        helper = load(path, name)
        call = helper._call

        def observed_score(*args, **kwargs):
            for arm in study.ARMS:
                trained = json.loads((output / f'{arm}_training.json').read_text())
                assert trained['state']['evaluation'] == []
            scored.append(args[0])
            return call(*args, **kwargs)

        helper._call = observed_score
        return helper

    monkeypatch.setattr(study, '_load', observed_load)
    case = study.run_study({'seed': plan['seed'], 'source_case': str(source)}, output,
        runtime=study.SOURCE / 'frozen_source')
    assert scored == ['score', 'score']
    assert not list(output.glob('*training.json'))
    assert not list(output.glob('*scored.json'))
    retained, removed = (case['results'][arm] for arm in study.ARMS)
    assert set(retained['initial_concept_ids']) - set(removed['initial_concept_ids']) == set(case['producer_added_concepts'])
    for arm in study.ARMS:
        report = case['results'][arm]
        for key in study.COST_KEYS:
            task_cost = sum(task['costs'][key] for task in report['tasks'])
            task_cost += report['screening_compute_cost'] if key == 'compute_cost' else 0
            assert report['task_sequence_costs'][key] == task_cost
            assert report['costs'][key] == task_cost + case['common_prefix_costs'][key]
            assert report['incremental_costs'][key] + case['fork_costs'][key] == task_cost
            assert report['task_sequence_costs'][key] <= report['task_sequence_resource_limits'][key]
            assert report['full_evaluation_costs'][key] == report['evaluation_costs'][key] + case['common_prefix_evaluation_costs'][key]
    summary = study.summarize([case], [], {'bootstrap_seed': 1, 'bootstrap_samples': 20})
    assert summary['completed_cases'] == 1
    assert summary['totals']['retain_product']['stages']['incoming_demand']['tasks'] == 1
