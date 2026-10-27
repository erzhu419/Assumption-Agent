from copy import deepcopy
from dataclasses import replace
import importlib.util,json
from pathlib import Path
import pytest
from test_registered_native_author import knowledge,LocalLab,paid_prefix
from test_native_process_study import case_plan,job_keys

@pytest.fixture
def isolated(tmp_path,monkeypatch):
    from scripts import run_anonymous_revision_inquiry as module
    root=tmp_path/'runtime'
    for name in ['src/hegel_machine/current_module.py','scripts/run_anonymous_revision_inquiry.py',
                 'scripts/run_online_inquiry_llm.py','scripts/run_online_inquiry_study.py']:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# toy runtime\n')
    monkeypatch.setattr(module,'ROOT',root)
    monkeypatch.setattr(module,'_aggregate_registered_sources',lambda *args: {'status':'toy_no_science'})
    return module

@pytest.fixture
def previous(tmp_path,monkeypatch,isolated):
    configured=case_plan(48);configured['runtime_snapshot']=[]
    def stop(agent,lab,path,**kwargs):
        agent._organization_state['finite_author']['completed_steps']=1
        agent.costs['compute_cost']=17;agent.costs['python_execution_seconds']=3.
        agent.relation_research['paid_fixture']={'cursor':{'next_binding':7},'completed_fit_ref':
            {'file':'shared/paid_fixture.json','json_path':[]}}
        (agent.workspace/'shared/paid_fixture.json').write_text('{"fit": "paid-original"}')
        agent._prepare_program_state()
        return {'status':'method_stopped'}
    monkeypatch.setattr(isolated,'execute_procedure',stop)
    path=tmp_path/'original'
    isolated.run_study(configured,path,knowledge=knowledge(),prepare_prefix=paid_prefix,
        laboratory_factory=lambda _:LocalLab(),workers=1)
    workspace=path/'case_001'/'fixed_pr2';checkpoint=workspace/'training.json'
    saved=json.loads(checkpoint.read_text());state=json.loads((workspace/'research_state.json').read_text())
    entry={'case':'case_001','arm':'fixed_pr2','checkpoint_ref':{'file':str(checkpoint),'json_path':[]},
       'completed_steps':1,'known_recorded_costs':saved['costs'],
       'central_state_recorded_costs':state['costs'],
       'unknown_cost_reservation':{'compute_cost':123,'python_execution_seconds':2.},
       'unrecorded_cost_basis_refs':[{'file':str(path/'stopped_operator_receipt.json'),'json_path':[]}]}
    (path/'stopped_operator_receipt.json').write_text('{"scope":"toy in-flight upper bound"}')
    return configured,path,entry

def test_registered_unfinished_training_resumes_only_selected_with_original_roster(
        tmp_path,monkeypatch,isolated,previous):
    configured,old,entry=previous;calls=[];rosters=[]
    def scheduler(train,jobs,workers):
        rosters.append(([(case['id'],arm) for case,arm in jobs],workers))
        return [train(job) for job in jobs]
    monkeypatch.setattr(isolated,'_native_process_trials',scheduler)
    def continued(agent,lab,path,**kwargs):
        calls.append((agent.workspace.parent.name,agent.workspace.name,kwargs['max_steps']))
        assert agent.costs['compute_cost']==17 and agent.costs['python_execution_seconds']==3.
        assert agent.costs['observation_cost']==1
        assert agent._organization_state['finite_author']['completed_steps']==1
        assert agent.relation_research['paid_fixture']['cursor']=={'next_binding':7}
        assert (agent.workspace/'shared/paid_fixture.json').read_text()=='{"fit": "paid-original"}'
        assert agent.config.compute_budget==configured['limits']['compute_budget']-123
        assert agent.config.execution_seconds==configured['limits']['execution_seconds']-2.
        assert lab.costs['observation_cost']==0
        agent.costs['compute_cost']+=5;agent.costs['python_execution_seconds']+=.25
        agent._organization_state['finite_author']['completed_steps']+=1
        agent._prepare_program_state()
        return {'status':'method_stopped','execution_timing':{'wall_seconds':.25}}
    monkeypatch.setattr(isolated,'execute_procedure',continued)
    out=tmp_path/'continued'
    result=isolated.run_study(configured,out,previous_study=old,workers=8,
        registered_continuation={'tasks':[entry]},laboratory_factory=lambda _:LocalLab(),
        prepare_prefix=lambda *args:pytest.fail('must not repurchase old evidence'))
    assert calls==[('case_001','fixed_pr2',47)]
    assert rosters==[(job_keys(configured),8)] and len(result['trials'])==9
    saved=json.loads((out/'case_001/fixed_pr2/training.json').read_text())
    assert saved['costs']['compute_cost']==22
    assert saved['continuation_cost']['compute_cost']==5
    assert saved['unknown_cost_reservation']==entry['unknown_cost_reservation']
    assert saved['registered_continuation_ceiling']['original_compute_budget']==configured['limits']['compute_budget']
    assert saved['execution_timing']=={'wall_seconds':.25}
    assert json.loads((old/'case_001/fixed_pr2/training.json').read_text())['costs']['compute_cost']==17
    assert all(trial['archive'].startswith(str(old)) for trial in result['trials']
               if (trial['case'],trial['arm'])!=('case_001','fixed_pr2'))

@pytest.mark.parametrize('field',['cost','steps','outside','unknown_basis'])
def test_changed_or_unbounded_checkpoint_is_not_restarted(tmp_path,monkeypatch,isolated,previous,field):
    configured,old,entry=previous;entry=deepcopy(entry)
    if field=='cost':entry['known_recorded_costs']['compute_cost']+=1
    elif field=='steps':entry['completed_steps']+=1
    elif field=='outside':entry['checkpoint_ref']['file']=str(old/'case_000/archive/training.json')
    else:entry['unrecorded_cost_basis_refs']=[]
    monkeypatch.setattr(isolated,'execute_procedure',lambda *a,**k:pytest.fail('invalid restore must not execute'))
    with pytest.raises(ValueError):
        isolated.run_study(configured,tmp_path/'invalid',previous_study=old,
            registered_continuation={'tasks':[entry]},laboratory_factory=lambda _:LocalLab())

@pytest.mark.parametrize('reason',['steps','budget','submitted'])
def test_submitted_or_no_original_allowance_reuses_saved_result(tmp_path,monkeypatch,isolated,previous,reason):
    configured,old,entry=previous;entry=deepcopy(entry)
    p=Path(entry['checkpoint_ref']['file']);saved=json.loads(p.read_text())
    if reason=='steps':saved['research_organization']['finite_author']['completed_steps']=48;entry['completed_steps']=48
    elif reason=='budget':entry['unknown_cost_reservation']['compute_cost']=configured['limits']['compute_budget']-17
    else:saved['model']='old.py'
    p.write_text(json.dumps(saved))
    prior,path,state,effective=isolated._registered_resume_checkpoint(entry,p.parent,
        isolated.InquiryLLMConfig(**configured['limits']),configured['author'])
    assert effective is None and state is None and path==p

@pytest.mark.parametrize('acquisition',['observation_cost','measurement_cost','model_calls'])
def test_new_acquisition_not_in_result_is_not_replayed(isolated,previous,acquisition):
    configured,old,entry=previous
    path=old/'case_001/fixed_pr2/research_state.json'
    state=json.loads(path.read_text());state['costs'][acquisition]+=1
    path.write_text(json.dumps(state));entry['central_state_recorded_costs']=state['costs']
    with pytest.raises(ValueError,match='paid acquisitions absent'):
        isolated._registered_resume_checkpoint(entry,path.parent,
            isolated.InquiryLLMConfig(**configured['limits']),configured['author'])


def test_newer_committed_fit_cost_is_retained_without_fresh_budget(isolated,previous):
    configured,old,entry=previous
    path=old/'case_001/fixed_pr2/research_state.json'
    state=json.loads(path.read_text());state['costs']['compute_cost']+=9
    state['relation_research']['paid_fixture']['cursor']['next_binding']=8
    path.write_text(json.dumps(state));entry['central_state_recorded_costs']=state['costs']
    result,_,retained,effective=isolated._registered_resume_checkpoint(entry,path.parent,
        isolated.InquiryLLMConfig(**configured['limits']),configured['author'])
    assert result['costs']['compute_cost']==26
    assert retained['relation_research']['paid_fixture']['cursor']['next_binding']==8
    assert effective.compute_budget==configured['limits']['compute_budget']-123


def test_active_checkpoint_can_continue_without_deleting_existing_training(
        tmp_path,monkeypatch,isolated,previous):
    configured,old,entry=previous
    workspace=old/'case_001/fixed_pr2'
    entry['checkpoint_ref']['file']=str(workspace/'active.json')
    called=[]
    def stop(agent,lab,path,**kwargs):
        called.append(kwargs['max_steps'])
        assert agent.costs['observation_cost']==1
        assert agent.relation_research['paid_fixture']['cursor']['next_binding']==7
        return {'status':'method_stopped'}
    monkeypatch.setattr(isolated,'execute_procedure',stop)
    isolated.run_study(configured,tmp_path/'active_resume',previous_study=old,
        registered_continuation={'tasks':[entry]},laboratory_factory=lambda _:LocalLab())
    assert called==[47]
    assert (workspace/'training.json').is_file()


def test_stopped_paid_receipts_reconcile_only_the_new_recovery_account(isolated, previous):
    configured, old, entry = previous
    path = Path(entry['checkpoint_ref']['file'])
    original = path.read_bytes()
    original_state = (path.parent / 'research_state.json').read_bytes()
    entry['known_uncommitted_costs'] = {'compute_cost': 9, 'python_execution_seconds': .75}
    entry['known_uncommitted_cost_refs'] = entry['unrecorded_cost_basis_refs']
    restored, _, state, effective = isolated._registered_resume_checkpoint(entry, path.parent,
        isolated.InquiryLLMConfig(**configured['limits']), configured['author'])
    assert restored['costs']['compute_cost'] == state['costs']['compute_cost'] == 26
    assert restored['costs']['python_execution_seconds'] == state['costs']['python_execution_seconds'] == 3.75
    assert restored['costs']['observation_cost'] == 1
    assert effective.compute_budget == configured['limits']['compute_budget'] - 123
    assert effective.execution_seconds == configured['limits']['execution_seconds'] - 2.
    assert path.read_bytes() == original
    assert (path.parent / 'research_state.json').read_bytes() == original_state


def test_stopped_known_receipts_cannot_be_filled_without_source_references(isolated, previous):
    configured, old, entry = previous
    entry['known_uncommitted_costs'] = {'compute_cost': 9, 'python_execution_seconds': .75}
    with pytest.raises(ValueError, match='receipts'):
        isolated._registered_resume_checkpoint(entry, Path(entry['checkpoint_ref']['file']).parent,
            isolated.InquiryLLMConfig(**configured['limits']), configured['author'])
