from scripts.run_partial_role_parent import save, load
from scripts.run_position_method_successor import ARMS, close, position_usage, protocol


def original():
    return {'tasks': [{'seed': 106110602, 'objects': 5, 'apparatus_seed': 106110602,
                       'name_seed': 22, 'wiring_seed': 33, 'relation_radius': 1., 'actuator_curvature': .2}],
        'limits': {'compute_cost': 300000, 'observation_cost': 320, 'measurement_cost': 6144},
        'activity_seconds': 1800, 'arms': ARMS, 'terminal': {'length': 48, 'trajectories': 1,
            'random_backgrounds': 0, 'sweep_fractions': [0., .25, .5, .75, 1.], 'mse_tolerance': 1e-7},
        'research_config': {'observation_budget': 320, 'compute_budget': 300000,
                            'minimum_improvement': 1e-7, 'max_candidates': 4}}


def test_position_source_quote_preserves_original_terminal_and_changes_physical_source():
    prior = original()
    result = protocol(prior)
    target, = result['tasks']
    assert (target['seed'], target['apparatus_seed']) == (106110603, 106110603)
    assert target['name_seed'] != prior['tasks'][0]['name_seed']
    assert result['terminal'] == prior['terminal'] and result['arms'] == ARMS
    assert result['budget_quote']['total_new_compute_upper'] == 600624
    assert result['budget_quote']['total_new_observation_upper'] == 1184


def test_closure_keeps_failure_interval_and_charges_shared_panel_once(tmp_path):
    plan = protocol(original())
    prior = tmp_path / 'prior.json'
    before = {'abc_remaining_compute_interval': [677356, 677356], 'abc_allocation_limit':1548741,
              'abc_remaining_observations':1253, 'cumulative_protocol_observations':6474,
              'cumulative_protocol_compute_interval':[4010076,5450666], 'complete_relation_unit_remaining':5243}
    save(prior, before)
    method = tmp_path / 'method.json'
    save(method, {'model_id': 'pos', 'source_ref': {'file': 'source'}})
    save(tmp_path / 'registration.json', {**plan, 'budget_contract_ref':{'file': str(prior)},
         'position_method_source': {'method_ref': {'file': str(method)}}})
    terminal = {'arms':{}, 'shared_actual_costs':{'compute_cost':416,'observation_cost':208,'measurement_cost':3120}}
    for arm in ARMS:
        row={'arm':arm,'frozen':True, 'status':'completed', 'costs':{'compute_cost':90,'observation_cost':30,'measurement_cost':450}}
        if arm=='pr3':
            row.update(status='failed',confirmed_compute_lower_bound=10)
            row['costs']['compute_cost']=None
        terminal['arms'][arm]={'prediction_measured':arm!='pr3'}
        save(tmp_path / 'runs/106110603' / arm / 'summary.json', row)
    save(tmp_path/'summary.json', {'paired_terminal':[terminal]})
    receipt=close(tmp_path); ledger=load(tmp_path/'closed_source_costs.json')
    assert receipt['new_cost_intervals']['compute_cost']==[606,200596]
    assert ledger['new_observations']==298 and ledger['abc_remaining_observations']==955
    assert ledger['abc_actual_compute_spent_to_date'] is None
    assert ledger['complete_relation_unit_remaining']==before['complete_relation_unit_remaining']
    assert load(prior)==before


def test_target_readout_separates_source_reconstruction_and_return_from_acquisition(tmp_path):
    method_path = tmp_path / 'method.json'
    source = {'method_ref': {'file': str(method_path)}}
    donor = {'file': 'paid_602'}
    save(method_path, {'model_id': 'learned', 'source_ref': donor})
    assert position_usage(tmp_path,'fixed_pr2',source)['purchased_rows'] is None
    parent = tmp_path / 'common'
    events = [
        {'action':'reconstruct_paid_object_method', 'source_ref':donor, 'method_available':True},
        {'action':'test_object_dependency_question', 'probe_id':'other', 'question':{'position_experiment':True},
         'observation_rows':1},
        {'action':'test_object_dependency_question', 'probe_id':'new', 'question':{'position_experiment':True,
         'source_object_model_id':'learned'}, 'observation_rows':1}]
    save(parent/'failed_paid_state.json',{'actions':events})
    result={'position_method_application':{'source_ref':donor}, 'position_first_return_ref':{'file':'first'},
            'position_experiment':{'field':'pos','excited_quantity':'q','baseline_ref':{'probe_id':'p'}}}
    save(parent/'object_question_000/consumption_001.json',result)
    save(parent/'object_question_000/consumption_002.json',
         {**result,'completed_relation':{'status':'position_domain_narrowed','field':'pos','retained_owners':['q2']}})
    usage = position_usage(tmp_path, 'fixed_pr2', source)
    assert usage['successful_source_reconstructions']==1 and not usage['frozen_method_supplied']
    assert usage['bound_target_questions']==1 and usage['purchased_rows']==1 and usage['narrowed_by_method']==1
