"""A partial paid instrument repair must leave the response research unresolved."""

from copy import deepcopy

from hegel_machine.cognition.competing_object_experiments import assess_competing_object_probe
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.object_continuation import decode_entry
from hegel_machine.cognition.relation_input_diagnosis import diagnose_relation_inputs
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_cognition_calibration_scope_revision import measured_range_source
from test_cognition_instrument_relation_diagnosis import SETTINGS, acquire, plan
from test_cognition_relation_input_recovery import PaidInputs, joint_probe


def test_budget_limited_mapping_does_not_execute_unbound_measured_fields(measured_range_source):
    knowledge, payload, source_batches = measured_range_source
    with UnifiedObjectLab(**SETTINGS, relation_condition='source_less') as laboratory:
        agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7),
            library=KnowledgeLibrary.from_dict(deepcopy(knowledge)))
        agent._learn_instruments = agent._revise_inputs = True
        selected = decode_entry(payload)
        original, priors = deepcopy(selected), deepcopy(agent.library.meta_priors)
        spec = laboratory.spec_for(('fine',))
        probe = joint_probe('already_paid_mechanism_experiment')
        paired, _ = acquire(laboratory, probe)
        old_binding = agent._instrument_binding(spec, laboratory.spec)
        forecast = plan(probe, selected['candidate'].program, old_binding)
        forecast['hypotheses'][0]['candidate_id'] = selected['candidate'].candidate_id
        evidence = assess_competing_object_probe(forecast, paired, SearchBudget(1000))
        assert evidence['calibration_status'] == 'counterexample'
        assert evidence['hypotheses'][0]['measured_loss'] < 1e-7

        paid = PaidInputs(laboratory, [*source_batches, paired])
        before = deepcopy(laboratory.costs)
        # This real allowance fits some instrument fields but cannot resolve
        # every field needed to forecast the measured response in commands.
        budget, actions = SearchBudget(2000), []
        result = diagnose_relation_inputs(agent, laboratory, selected, selected,
            forecast, evidence, paid.acquire, paid.paired_for, budget, actions, channels=('fine',))
        binding = agent._instrument_binding(spec, laboratory.spec)
        assert selected['candidate'].program.required_features() - set(binding)
        assert result['status'] == 'unresolved_inputs' and result['entry'] is None
        assert result['response_status'] == 'unassessed'
        assert 'repaired_evidence' not in result
        assert result['binding'] == old_binding
        assert selected == original and agent.library.meta_priors == priors
        assert 'remeasure_relation_counterexample' not in paid.purposes
        assert laboratory.costs['observation_cost'] - before['observation_cost'] == paid.rows
        rate = next(channel.cost_per_row for channel in laboratory.channels if channel.name == 'fine')
        assert laboratory.costs['measurement_cost'] - before['measurement_cost'] == rate * paid.rows
        assert 0 < budget.spent <= budget.limit
