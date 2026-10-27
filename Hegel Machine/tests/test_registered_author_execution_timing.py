from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from test_registered_native_author import knowledge,LocalLab,plan


def test_real_procedure_outcome_timing_survives_training_trial_and_recovery(tmp_path,monkeypatch):
    from scripts import run_anonymous_revision_inquiry as study
    runtime=tmp_path/'runtime'
    for name in ['src/hegel_machine/current_module.py','scripts/run_anonymous_revision_inquiry.py',
                 'scripts/run_online_inquiry_llm.py','scripts/run_online_inquiry_study.py']:
        p=runtime/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# isolated control\n')
    monkeypatch.setattr(study,'ROOT',runtime)
    timing={'wall_seconds':2.5,'host_cpu_seconds':1.75,
        'phase_seconds':{'author':.25,'action':1.5,'before':.75},
        'scope':'existing_execution_timing_fixture'}
    monkeypatch.setattr(study,'execute_procedure',lambda *a,**k:{'status':'method_stopped',
        'execution_timing':deepcopy(timing)})
    configured=plan(steps=1);configured['runtime_snapshot']=[]
    output=tmp_path/'study'
    summary=study.run_study(configured,output,knowledge=knowledge(),
        prepare_prefix=lambda *a:{},laboratory_factory=lambda _:LocalLab(),workers=1)
    assert len(summary['trials'])==3
    for trial in summary['trials']:
        assert trial['execution_timing']==timing
        saved=json.loads((output/trial['case']/trial['arm']/'training.json').read_text())
        assert saved['execution_timing']==timing
        assert saved['costs']['python_execution_seconds']==0
        assert saved['costs']['compute_cost']==saved['costs']['observation_cost']==0
        replay=study._saved_study_trial({'id':trial['case']},trial['arm'],saved,
            output/trial['case']/trial['arm']/'training.json',0)
        assert replay['result_summary']['execution_timing']==timing
    assert 'execution_timing' in study.STUDY_RESULT_FIELDS
