import json
from random import Random

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.structural_discovery import StructuralDiscovery
from hegel_machine.cognition.structure_lab import StructureLab
from hegel_machine.cognition.unified_structure import propose_structures
from hegel_machine.cognition.unified_uncertainty import assess_noisy_adequacy


def test_missing_object_window_is_a_structural_counterexample_not_noise():
    lab = StructureLab(readout="window", seed=71)
    training = [row for seed in (89, 129)
                for row in lab.acquire(lab.random_probe(seed, length=32)).observations]
    validation = lab.acquire(lab.random_probe(907, length=48)).observations
    incumbent = Program("add", children=(Program("const", value=.05),
        Program("mul", children=(Program("const", value=.3), Program("feature", name="q_o0")))))
    result = assess_noisy_adequacy(lab.spec, incumbent, training, validation, SearchBudget(100000))
    assert not result["adequate"] and result["reason"] == "predictable_residual"
    counterexample = result["structural_counterexample"]
    assert counterexample["comparison_basis"] == "direct_target"
    assert counterexample["kind"] == "structural:window_exchange"
    assert counterexample["gain"] > counterexample["margin"]
    correction = Program.from_dict(json.loads(json.dumps(counterexample["program"])))
    composed = Program("add", children=(incumbent, correction))
    assert loss(composed, validation, 0., .5) < .01 * loss(incumbent, validation, 0., .5)
    assert "window_sum" in composed.describe()


def test_actual_structural_budget_truncation_survives_serialization():
    lab = StructureLab(seed=71)
    training = lab.acquire(lab.random_probe(89, length=24)).observations
    partial = propose_structures(lab.spec, training, SearchBudget(24 * 4))
    complete = propose_structures(lab.spec, training, SearchBudget(80000))
    assert partial and not partial[0]["structure"]["search_complete"]
    assert complete[0]["structure"]["search_complete"]
    restored = StructuralDiscovery.from_dict(json.loads(json.dumps(partial[0]["structure"])))
    assert not restored.search_complete


def test_public_object_schema_still_allows_supported_squared_loss_noise():
    lab = StructureLab(readout="total", seed=71)
    spec = lab.spec_for(scale=4)
    schema = lab.schema_for(scale=4)
    fields = schema.fields[schema.objects[0]]
    program = Program("add", children=(Program("feature", name=fields["q"]),
                                       Program("feature", name=fields["u"])))

    def trajectory(seed):
        rng, rows, history = Random(seed), [], []
        batch = lab.acquire(lab.random_probe(seed, length=32), scale=4)
        for row in batch.observations:
            target = row.target + rng.uniform(-.1, .1)
            rows.append(Observation(row.features, target, tuple(history)))
            history.append({**row.features, "__target__": target})
        return rows

    training, validation = trajectory(89) + trajectory(129), trajectory(907) + trajectory(997)
    result = assess_noisy_adequacy(spec, program, training, validation, SearchBudget(100000))
    assert result["adequate"] and result["reason"] == "noise_compatible"
    assert result["fixed_quantile"] == .5
    assert any(item.get("comparison_basis") == "direct_target" for item in result["residual_comparisons"])
    limited = assess_noisy_adequacy(spec, program, training, validation, SearchBudget(3000))
    assert not limited["adequate"] and limited["reason"] == "insufficient_budget"
