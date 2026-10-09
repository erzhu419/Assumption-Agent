"""Experimental-contract checks, separate from the registered scientific cohort."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.online_inquiry import InquiryLab, public_program


PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'run_online_inquiry_study.py'
SPEC = importlib.util.spec_from_file_location('online_inquiry_study', PATH)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_registered_cohorts_are_disjoint_and_change_structure_and_binding():
    development = study.cohort_protocol('development')
    main = study.cohort_protocol('main')
    assert len(development['cases']) == 6
    assert len(main['cases']) == 30
    assert not ({case['seed'] for case in development['cases']} &
                {case['seed'] for case in main['cases']})
    for case in main['cases']:
        source = case['source']['lab']
        assert len(case['consumers']) == 4
        assert {item['condition'] for item in case['consumers']} == set(study.CONDITIONS)
        for item in case['consumers']:
            target = item['lab']
            assert target['seed'] == source['seed']
            assert target['driver'] == source['driver']
            assert target['orientation'] == source['orientation']
            assert (target['mechanism'] == source['mechanism']) == item['same_structure']
            assert (target['surface'] == source['surface']) == item['same_surface']
            if item['same_surface']:
                assert target['calibration_sign'] == source['calibration_sign']
                assert target['calibration_offset'] == source['calibration_offset']
            else:
                assert target['calibration_sign'] == -source['calibration_sign']
                assert target['calibration_offset'] != source['calibration_offset']
                assert study.make_lab(source).public_spec()['controls'] != \
                    study.make_lab(target).public_spec()['controls']


def test_terminal_schema_separates_equivalent_law_and_false_confirmation():
    lab = InquiryLab(seed=81, mechanism='quadratic', driver=0, orientation=1)
    # Sign is not identifiable for a square; exact parameter matching is wrong.
    equivalent = public_program({'mechanism': 'quadratic', 'driver': 0, 'orientation': -1},
                                lab.public_spec())
    record = study.evaluate_record({'program': equivalent.to_dict(), 'confirmed': True},
                                   lab, evaluation_seed=982)
    assert record['metrics']['correct_law'] is True
    assert record['metrics']['confirmed_and_correct'] is True
    wrong = study.evaluate_record({'program': Program('const', value=0.).to_dict(),
                                   'confirmed': True}, lab, evaluation_seed=982)
    assert wrong['metrics']['hypothesis_executable'] is True
    assert wrong['metrics']['false_confirmation'] is True
    assert wrong['metrics']['correct_law'] is None
    assert lab.observation_cost == 0  # terminal rows are never training observations


def test_child_process_metric_requires_online_resolution_not_just_node_presence():
    report = {'nodes': [{'question_id': 'q1', 'parent_id': 'q0',
                        'opened_at': 3, 'resolved_at': 2, 'status': 'resolved'}],
              'events': [{'event': 'parent_resumed', 'consumes_answer': 'q1'}]}
    lab = InquiryLab()
    metrics = study.evaluate_record(report, lab, evaluation_seed=81)['metrics']
    assert not metrics['online_subresearch_completed']
    assert metrics['consumed_child_answers'] == 0
    report['nodes'][0]['resolved_at'] = 7
    metrics = study.evaluate_record(report, lab, evaluation_seed=81)['metrics']
    assert metrics['online_subresearch_completed']
    assert metrics['consumed_child_answers'] == 1


def test_case_freezes_shared_source_and_finishes_research_before_terminal_evaluation(tmp_path, monkeypatch):
    plan = study.cohort_protocol('development')
    case = plan['cases'][0]
    original_train, original_evaluate = study._train, InquiryLab.blind_evaluate
    trained = []
    source_checkpoint = None

    def observed_train(spec, *, mode, state, observation_budget, compute_budget):
        nonlocal source_checkpoint
        before = deepcopy(state)
        if trained:
            assert state == source_checkpoint
        result = original_train(spec, mode=mode, state=state, observation_budget=observation_budget,
                                compute_budget=compute_budget)
        assert state == before  # no sibling can mutate the common source checkpoint
        trained.append(result[0]['research'])
        if len(trained) == 1:
            source_checkpoint = deepcopy(result[1])
        return result

    def observed_evaluate(self, *args, **kwargs):
        assert len(trained) == 1 + len(study.ARMS) * len(study.CONDITIONS)
        return original_evaluate(self, *args, **kwargs)

    monkeypatch.setattr(study, '_train', observed_train)
    monkeypatch.setattr(InquiryLab, 'blind_evaluate', observed_evaluate)
    result = study.run_case(case, plan, tmp_path / 'chain')
    assert len(result['results']) == 12
    assert all(row['status'] == 'completed' for row in result['results'])
    assert all(row['evaluation_error'] is None for row in result['results'])
    assert all(row['research']['costs']['observation_cost'] <= plan['observation_budget']
               for row in result['results'])
    assert all(row['research']['costs']['compute_cost'] <= plan['compute_budget']
               for row in result['results'])
    assert json.loads((tmp_path / 'chain' / 'source_knowledge.json').read_text()) == source_checkpoint
    assert result['source']['metrics']['confirmed_and_correct']
    assert {row['arm'] for row in result['results']} == set(study.ARMS)


def test_summary_keeps_failed_chains_in_denominator():
    plan = study.cohort_protocol('development')
    registered = plan['cases'][0]
    rows = [{'arm': arm, 'condition': condition, 'status': 'completed',
             'research': {'costs': {'observation_cost': 2}},
             'metrics': {'confirmed_and_correct': True, 'correct_law': True}}
            for arm in plan['arms'] for condition in study.CONDITIONS]
    case = {'chain_id': registered['chain_id'], 'seed': registered['seed'], 'results': rows,
            'source': {'metrics': {'confirmed_and_correct': True},
                       'research': {'costs': {'observation_cost': 3}}}}
    failures = [{'chain_id': row['chain_id'], 'error': 'source acquisition failed'}
                for row in plan['cases'][1:]]
    summary = study.summarize([case], failures, plan)
    for arm in plan['arms']:
        total = summary['totals'][arm]
        assert total['metrics']['confirmed_and_correct']['count'] == 4
        assert total['metrics']['confirmed_and_correct']['registered_tasks'] == 24
        assert total['metrics']['confirmed_and_correct']['rate_over_registered'] == 1 / 6
        assert total['chain_costs_including_source_once']['observation_cost'] == 11
    for cells in summary['paired_continuations'].values():
        assert all(cell['both_correct'] == 1 and cell['missing_pair'] == 5 for cell in cells.values())
