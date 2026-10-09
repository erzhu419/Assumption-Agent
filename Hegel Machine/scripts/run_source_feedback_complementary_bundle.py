#!/usr/bin/env python3
"""Measure a genuinely complementary two-source executable bundle.

Source A contains the current measurement and source B contains a lagged
measurement.  The joint candidate calls both learned definitions, so the
source association is present in the executable program rather than only in
metadata.  Four value masks use one held-out response.  A second mechanism
then records a counterexample for the joint binding and checks that the two
single-source calls remain reusable.
"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "source_feedback_complementary_bundle_20261004"
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, TaskSpec
from hegel_machine.cognition.knowledge_returns import transfer_condition


SEED = 20261010
SOURCE_A = "concept_0001"
SOURCE_B = "concept_0002"
LINEAGE = "joint:complementary:current_plus_lag"
MASKS = ("keep_ab", "drop_a", "drop_b", "drop_ab")


def _rows(seed, count, current_weight, lag_weight):
    rng = Random(seed)
    history = []
    rows = []
    for _ in range(count):
        x = rng.uniform(-1.0, 1.0)
        features = {"x": x}
        lag = history[-2]["x"] if len(history) >= 2 else 0.0
        rows.append(Observation(features, current_weight * x + lag_weight * lag,
                                tuple(dict(item) for item in history)))
        history.append(features)
    return rows


def _spec(task_id):
    return TaskSpec(task_id, "predict a response using current and lagged measurements",
                    ("x",), (), (), feature_bounds={"x": (-1.0, 1.0)})


def _source_programs():
    x = Program("feature", name="x")
    lag = Program("lag", name="x", lag=2)
    available = Program("available", lag=2)
    a = Program("mul", children=(Program("const", value=.6), x))
    b = Program("if", children=(available,
        Program("mul", children=(Program("const", value=1.2), lag)),
        Program("const", value=0.0)))
    return a, b


def _calls(a_id, b_id, a_body, b_body):
    a_def = Definition("source_a_current", (), a_body, source_concept_id=a_id)
    b_def = Definition("source_b_lag", (), b_body, source_concept_id=b_id)
    a_call = Program("call", name=a_def.definition_id, definition=a_def)
    b_call = Program("call", name=b_def.definition_id, definition=b_def)
    return a_call, b_call


def _library(source_rows):
    library = KnowledgeLibrary()
    a_body, b_body = _source_programs()
    a_id, _ = library.learn(Candidate("learn_a", a_body, "current_term"),
                            _spec("source_a"), observations=source_rows)
    b_id, _ = library.learn(Candidate("learn_b", b_body, "lag_term"),
                            _spec("source_b"), observations=source_rows)
    if (a_id, b_id) != (SOURCE_A, SOURCE_B):
        raise RuntimeError(f"unexpected source ids: {(a_id, b_id)}")
    return library, a_body, b_body


def _origin(source_id, template_id, condition):
    return {"source_id": source_id, "template_id": template_id, "mapping": {},
            "mapping_roles": {}, "condition": deepcopy(condition),
            "lineage_id": LINEAGE, "joint_source_ids": [SOURCE_A, SOURCE_B],
            "adaptation_work": [{"work_id": "bundle:shared_adaptation",
                "compute_cost": 24, "source_ids": [SOURCE_A, SOURCE_B],
                "template_ids": [template_id], "preserved_work": []}]}


def _joint_transfer(library, spec, rows, candidate, a_template, b_template,
                    baseline_loss, joint_loss):
    library.begin_transfer_episode(spec, rows)
    condition = transfer_condition(spec, rows)
    for source_id, template_id in ((SOURCE_A, a_template), (SOURCE_B, b_template)):
        origin = _origin(source_id, template_id, condition)
        local = replace(candidate, candidate_id=f"joint:{source_id}",
                        memory_id=source_id)
        library.record_transfer_result(local, spec, accepted=True, loss=joint_loss,
            baseline_loss=baseline_loss, evidence_rows=len(rows),
            comparison_compute=48, comparison_id="joint-comparison", origin=origin)
    return library.finish_transfer_episode(spec, final_candidate=candidate,
        final_loss=joint_loss, baseline_loss=baseline_loss, compute_cost=72,
        observation_cost=len(rows), validation_rows=len(rows), final_status="completed")


def _value(mask, rows, heldout, library, candidates, baseline):
    if mask == "keep_ab":
        program = candidates["ab"]
    elif mask == "drop_a":
        program = candidates["b"]
    elif mask == "drop_b":
        program = candidates["a"]
    else:
        return {"heldout_loss": baseline, "value": 0.0,
                "program": None, "selected_sources": []}
    fallback = 0.0
    measured = loss(program.program, heldout, fallback, .5)
    return {"heldout_loss": measured, "value": baseline - measured,
            "program": program.program.to_dict(),
            "selected_sources": list(program.parents)}


def main():
    rows = _rows(SEED, 48, .6, 1.2)
    heldout = _rows(SEED + 101, 64, .6, 1.2)
    source_rows = _rows(SEED + 202, 48, .6, 1.2)
    library, a_body, b_body = _library(source_rows)
    a_template = next(item.template_id for item in library.abstracts
                       if item.source_concept_id == SOURCE_A)
    b_template = next(item.template_id for item in library.abstracts
                       if item.source_concept_id == SOURCE_B)
    a_call, b_call = _calls(SOURCE_A, SOURCE_B, a_body, b_body)
    a_candidate = Candidate("single_a", a_call, "single_source", parents=(SOURCE_A,))
    b_candidate = Candidate("single_b", b_call, "single_source", parents=(SOURCE_B,))
    ab_program = Program("add", children=(a_call, b_call))
    ab_candidate = Candidate("joint_ab", ab_program, "joint_bundle",
                             parents=(SOURCE_A, SOURCE_B))
    baseline = loss(Program("const", value=0.0), rows, 0.0, .5)
    a_loss = loss(a_call, rows, 0.0, .5)
    b_loss = loss(b_call, rows, 0.0, .5)
    ab_loss = loss(ab_program, rows, 0.0, .5)
    returns = _joint_transfer(library, _spec("complementary_target"), rows,
                              ab_candidate, a_template, b_template, baseline, ab_loss)
    values = {mask: _value(mask, rows, heldout, library,
                            {"a": a_candidate, "b": b_candidate, "ab": ab_candidate},
                            loss(Program("const", value=0.0), heldout, 0.0, .5))
              for mask in MASKS}
    v_ab, v_a, v_b, v_empty = (values["keep_ab"]["value"], values["drop_b"]["value"],
                               values["drop_a"]["value"], values["drop_ab"]["value"])
    contrast = {"delta_a": v_ab - v_b, "delta_b": v_ab - v_a,
                "interaction": v_ab - v_a - v_b + v_empty}
    if not (ab_loss < 1e-20 and a_loss > 0.01 and b_loss > 0.01):
        raise RuntimeError(f"the source terms are not complementary: {a_loss}, {b_loss}, {ab_loss}")
    if not (contrast["delta_a"] > 0.1 and contrast["delta_b"] > 0.1
            and v_ab > max(v_a, v_b) + 0.1):
        raise RuntimeError(f"joint value did not show complementarity: {contrast}")
    if len(returns["joint_returns"]) != 1 or not returns["joint_returns"][0]["shared_cost_accounted_once"]:
        raise RuntimeError("joint return did not deduplicate shared work")

    shifted = _rows(SEED + 303, 48, .6, .45)
    shifted_spec = _spec("mechanism_shift")
    changed_joint_loss = loss(ab_program, shifted, 0.0, .5)
    changed_a_loss = loss(a_call, shifted, 0.0, .5)
    changed_b_loss = loss(b_call, shifted, 0.0, .5)
    withdrawal = library.record_transfer_counterexample(shifted_spec, ab_candidate,
        [SOURCE_A, SOURCE_B], probe_id="mechanism-shift-1",
        evidence={"status": "counterexample", "rows": len(shifted),
                  "measured_loss": changed_joint_loss,
                  "retained_loss": loss(Program("const", value=0.0), shifted, 0.0, .5),
                  "lineage_id": LINEAGE})
    if withdrawal is None:
        raise RuntimeError("joint mechanism counterexample was not recorded")
    if not library.transfer_refuted(shifted_spec, ab_program):
        raise RuntimeError("joint binding was not withdrawn")
    if library.transfer_refuted(shifted_spec, a_call) or library.transfer_refuted(shifted_spec, b_call):
        raise RuntimeError("joint withdrawal incorrectly revoked a single source")
    payload = {"status": "complete", "lineage": {"package_version": "0.181.0",
        "parent": "joint source lineage and shared-cost attribution",
        "stage": "complementary current-term plus lag-term bundle"},
        "protocol": {"seed": SEED, "masks": list(MASKS), "same_paid_rows": True,
            "external_model_calls": 0, "source_ids": {"a": SOURCE_A, "b": SOURCE_B},
            "lineage_id": LINEAGE},
        "source_programs": {"a": a_body.describe(), "b": b_body.describe(),
            "joint": ab_program.describe()},
        "training_losses": {"baseline": baseline, "a": a_loss, "b": b_loss, "ab": ab_loss},
        "values": values, "contrast": contrast,
        "joint_returns": returns["joint_returns"],
        "mechanism_shift": {"joint_loss": changed_joint_loss, "a_loss": changed_a_loss,
            "b_loss": changed_b_loss, "withdrawal": withdrawal,
            "joint_refuted": library.transfer_refuted(shifted_spec, ab_program),
            "a_refuted": library.transfer_refuted(shifted_spec, a_call),
            "b_refuted": library.transfer_refuted(shifted_spec, b_call)},
        "interpretation": "Current and lagged terms each add positive value beyond the other singleton; finite-sample interaction is reported rather than used as a synergy gate, and a mechanism-shift counterexample withdraws only their joint binding.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "contrast": contrast,
        "output": str((OUT / "results.json").relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
