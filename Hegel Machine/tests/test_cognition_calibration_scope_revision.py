"""Repair a failed input forecast without discarding valid measured range evidence."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.competing_object_experiments import assess_competing_object_probe
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.mechanism_revision import propose_mechanism_revisions
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.object_continuation import decode_entry, encode_entry
from hegel_machine.cognition.program_experiment_methods import _failure_scope, _same_failed_field
from hegel_machine.cognition.relation_input_diagnosis import (
    bind_relation_entry, diagnose_relation_inputs, measured_relation_entry,
)
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_cognition_instrument_relation_diagnosis import SETTINGS, acquire, plan
from test_cognition_relation_input_recovery import PaidInputs, command_evidence, joint_probe, setup_source


@pytest.fixture(scope='module')
def measured_range_source():
    with UnifiedObjectLab(**SETTINGS, relation_condition='source_less') as laboratory:
        agent, origin, batches = setup_source(laboratory)
        spec, budget = laboratory.spec_for(('fine',)), SearchBudget(100000)
        measured = bind_relation_entry(measured_relation_entry(origin, spec, agent.library),
            {name: Program('feature', name=name) for name in spec.feature_names})
        rows = [row for batch in batches for _, row in batch]
        research = RelationIntervalResearch(measured['candidate'].program, agent.library.meta_priors,
            budget, research_program=measured['research_program'])
        alternatives = research.alternatives(rows, budget,
            experiments=([row for _, row in batches[-1]],))
        revised = propose_mechanism_revisions(spec, rows, budget, parent=alternatives[0],
            priors=agent.library.meta_priors)
        selected = next(entry for entry in revised if entry['kind'] == 'mechanism_full_refit')
        assert selected['training_loss'] < 1e-7
        scope = selected['research_program']['selection_steps'][-1]
        assert scope['evidence_contexts']
        assert all(Program.from_dict(context['parameter_template']).required_features()
                   <= set(spec.feature_names) for context in scope['evidence_contexts'])
        assert any(low < .8 < high for parameter in scope['parameters']
                   for low, high in parameter['radius_intervals'])
        return json.loads(json.dumps(agent.library.to_dict())), \
            json.loads(json.dumps(encode_entry(selected))), batches


@pytest.mark.parametrize('condition', ['source_less', 'source_greater'])
def test_paid_mechanism_calibration_repair_preserves_measured_scope_and_retests_the_same_trace(
        measured_range_source, condition):
    knowledge, payload, source_batches = measured_range_source
    with UnifiedObjectLab(**SETTINGS, relation_condition=condition) as laboratory:
        agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7),
                                     library=KnowledgeLibrary.from_dict(deepcopy(knowledge)))
        agent._learn_instruments = agent._revise_inputs = True
        selected = decode_entry(payload)
        original, definitions = deepcopy(selected), deepcopy(agent.library.definitions)
        priors = deepcopy(agent.library.meta_priors)
        spec = laboratory.spec_for(('fine',))
        probe = joint_probe('already_paid_mechanism_experiment')
        paired, _ = acquire(laboratory, probe)
        forecast = plan(probe, selected['candidate'].program,
            agent._instrument_binding(spec, laboratory.spec))
        forecast['hypotheses'][0]['candidate_id'] = selected['candidate'].candidate_id
        evidence = assess_competing_object_probe(forecast, paired, SearchBudget(1000))
        original_evidence = deepcopy(evidence)
        assert evidence['calibration_status'] == 'counterexample'
        if condition == 'source_less':
            assert evidence['hypotheses'][0]['measured_loss'] < 1e-7
            assert evidence['hypotheses'][0]['forecast_loss'] > 1e-7

        paid = PaidInputs(laboratory, [*source_batches, paired])
        before = deepcopy(laboratory.costs)
        budget, actions = SearchBudget(60000), []
        result = diagnose_relation_inputs(agent, laboratory, selected, selected,
            forecast, evidence, paid.acquire, paid.paired_for, budget, actions, channels=('fine',))
        assert result is not None and result['status'] == 'inputs_revised'
        assert 'remeasure_relation_counterexample' not in paid.purposes
        assert evidence == original_evidence and result['evidence']['calibration_status'] == 'counterexample'
        assert result['repaired_evidence']['calibration_status'] == 'supported_on_probe'
        assert result['repaired_plan']['probe'] == probe
        repaired = result['repaired_plan']['hypotheses'][0]
        for predicted, (_, measured) in zip(repaired['predictions'], paired):
            assert predicted == pytest.approx(selected['candidate'].program.predict(measured), abs=1e-8)

        # A new command-to-reading map does not invalidate facts purchased in
        # measured coordinates. Keep their ranges, targets, and old consumer.
        assert result['entry']['candidate'].program == selected['candidate'].program
        assert result['entry']['research_program'] == selected['research_program']
        assert selected == original and agent.library.definitions == definitions
        assert agent.library.meta_priors == priors
        assert result['response_rows'] and all(set(row.features) == set(spec.feature_names)
                                               for row in result['response_rows'])
        assert result['response_loss'] == pytest.approx(loss(result['entry']['candidate'].program,
            [*result['response_rows'], *(measured for _, measured in paired)], 0., spec.risk_quantile))
        expected = 'supported_on_paid_rows' if condition == 'source_less' else 'counterexample'
        assert result['response_status'] == expected
        expected_probe = 'supported_on_probe' if condition == 'source_less' else 'counterexample'
        assert result['repaired_evidence']['hypotheses'][0]['status'] == expected_probe
        assert len(result['response_rows']) == len(result['command_rows']) == paid.rows
        assert laboratory.costs['observation_cost'] - before['observation_cost'] == paid.rows
        rate = next(channel.cost_per_row for channel in laboratory.channels if channel.name == 'fine')
        assert laboratory.costs['measurement_cost'] - before['measurement_cost'] == rate * paid.rows
        assert 0 < budget.spent <= budget.limit


def test_command_failure_scope_keeps_the_refuted_mapping_after_input_repair():
    with UnifiedObjectLab(**SETTINGS, relation_condition='source_less') as laboratory:
        agent, origin, batches = setup_source(laboratory)
        spec = laboratory.spec_for(('fine',))
        before = agent._instrument_binding(spec, laboratory.spec)
        forecast, evidence = command_evidence(laboratory, origin,
                                               joint_probe('command_input_counterexample'))
        paid = PaidInputs(laboratory, batches)
        result = diagnose_relation_inputs(agent, laboratory, origin, origin, forecast, evidence,
            paid.acquire, paid.paired_for, SearchBudget(60000), [])
        assert result['status'] == 'inputs_revised'
        assert result['response_status'] == 'supported_on_paid_rows'
        assert result['evidence']['calibration_status'] == 'counterexample'
        assert 'repaired_evidence' not in result
        after = agent._instrument_binding(spec, laboratory.spec)
        assert result['binding'] == before and after != before
        scope = _failure_scope(result['plan'], result['evidence'], result['binding'],
                               laboratory.spec, 'repaired_command', 1e-7)
        assert scope['fields']
        assert all(_same_failed_field(field, before, laboratory.spec.feature_bounds)
                   for field in scope['fields'])
        assert all(not _same_failed_field(field, after, laboratory.spec.feature_bounds)
                   for field in scope['fields'])
