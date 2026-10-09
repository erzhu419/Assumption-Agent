"""Online, dependency-directed inquiry in a finite calibrated-response language.

Unlike a graph compiled from a completed trace, a question here is opened before
its observations are acquired. A response question may request a coordinate
binding, which may in turn request a sensor calibration. The returned answers
condition the parent's experiment. A flat active learner has the identical joint
candidate space, observations and experiment menu; it need not follow that graph.

This is a deliberately finite mechanism study, not an unrestricted proof system.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import product
import math
from random import Random

from .language import Definition, Program
from .models import Observation, TaskSpec


MECHANISMS = ("linear", "quadratic", "absolute", "threshold")
LEVELS = (-1., -.6, 0., .6, 1.)
TOLERANCE = 1e-10


def _law(mechanism, value):
    if mechanism == "linear":
        return value
    if mechanism == "quadratic":
        return value * value
    if mechanism == "absolute":
        return abs(value)
    if mechanism == "threshold":
        return float(value >= .35)
    raise ValueError("unknown finite response mechanism")


def _law_program(mechanism, z):
    if mechanism == "linear":
        return z
    if mechanism == "quadratic":
        return Program("mul", children=(z, z))
    if mechanism == "absolute":
        return Program("abs", children=(z,))
    if mechanism == "threshold":
        return Program("if", children=(Program("ge", children=(z, Program("const", value=.35))),
                                       Program("const", value=1.), Program("const", value=0.)))
    raise ValueError("unknown finite response mechanism")


def public_candidates(spec):
    """The public finite grammar, with no access to a laboratory's truth."""
    language = spec["finite_language"]
    return [dict(zip(("mechanism", "driver", "orientation", "calibration_sign",
                      "calibration_offset"), values))
            for values in product(language["mechanisms"], language["drivers"],
                                  language["orientations"], language["calibration_signs"],
                                  language["calibration_offsets"])]


def public_program(hypothesis, spec):
    """Compile a proposed hypothesis; this performs no fitting or selection."""
    z = Program("feature", name=spec["controls"][hypothesis["driver"]])
    if hypothesis["orientation"] == -1:
        z = Program("mul", children=(Program("const", value=-1.), z))
    return _law_program(hypothesis["mechanism"], z)


def predict_action(hypothesis, action, spec):
    """Public forward prediction used identically by the two search controls."""
    if action["kind"] == "calibrate":
        z = action["level"]
    else:
        z = hypothesis["orientation"] * action["point"][spec["controls"][hypothesis["driver"]]]
    if action["kind"] == "response":
        return _law(hypothesis["mechanism"], z)
    return hypothesis["calibration_sign"] * z + hypothesis["calibration_offset"]


class InquiryLab:
    """A deterministic process with an anonymous, uncalibrated instrument.

    The constructor specifies evaluator truth. The learner uses public_spec,
    available_actions and execute only. Surface 1 both renames and permutes the
    controls, so a stored positional binding cannot silently denote a true role.
    """

    def __init__(self, *, seed=1, surface=0, mechanism="linear", driver=None,
                 orientation=None, calibration_sign=None, calibration_offset=None):
        if mechanism not in MECHANISMS:
            raise ValueError("unknown mechanism")
        rng = Random(seed)
        physical_driver = rng.randrange(2) if driver is None else driver
        self._truth = {"mechanism": mechanism, "driver": (physical_driver + surface) % 2,
                       "orientation": rng.choice((-1, 1)) if orientation is None else orientation,
                       "calibration_sign": rng.choice((-1, 1)) if calibration_sign is None else calibration_sign,
                       "calibration_offset": rng.choice((-1, 0, 1)) if calibration_offset is None else calibration_offset}
        names = Random(seed + 1009 * surface).sample(range(100, 900), 2)
        self._controls = tuple(f"c{n}" for n in names)
        self.spec = TaskSpec("calibrated_response", "Identify and predict an unknown response.",
                             self._controls, (), (), feature_bounds={n: (-1., 1.) for n in self._controls})
        self.observations = []
        self.observation_cost = 0

    def public_spec(self):
        return {"controls": list(self._controls), "feature_bounds": dict(self.spec.feature_bounds),
                "action_kinds": ["response", "measure", "calibrate"],
                "finite_language": {"mechanisms": list(MECHANISMS), "drivers": [0, 1],
                    "orientations": [-1, 1], "calibration_signs": [-1, 1],
                    "calibration_offsets": [-1, 0, 1], "threshold": .35},
                "instrument": "readout", "cost_per_observation": 1}

    def available_actions(self):
        return ([{"kind": "calibrate", "level": level} for level in (-1., 0., 1.)] +
                [{"kind": kind, "point": dict(zip(self._controls, levels))}
                 for kind in ("measure", "response") for levels in product(LEVELS, repeat=2)])

    def action_cost(self, action):
        return 1

    def observation_rows(self, record):
        return ([Observation(record["action"]["point"], record["value"])]
                if record["action"]["kind"] == "response" else [])

    def confirmation_actions(self, count, rng):
        return [{"kind": "response", "point": {name: rng.uniform(-1., 1.)
                 for name in self._controls}} for _ in range(count)]

    def execute(self, action):
        if action["kind"] not in self.public_spec()["action_kinds"]:
            raise ValueError("unknown experiment")
        if action["kind"] == "calibrate":
            if not -1 <= action["level"] <= 1:
                raise ValueError("reference level outside supported range")
        elif (set(action["point"]) != set(self._controls) or
              any(not -1 <= x <= 1 for x in action["point"].values())):
            raise ValueError("supply both public controls within their bounds")
        value = predict_action(self._truth, action, self.public_spec())
        result = {"action": deepcopy(action), "value": value,
                  "observation_index": self.observation_cost}
        self.observations.append(result)
        self.observation_cost += 1
        return deepcopy(result)

    @property
    def costs(self):
        return {"observation_cost": self.observation_cost, "measurement_cost": 0,
                "queries": self.observation_cost}

    def evaluation_rows(self, *, length=256, seed=1000003):
        rng = Random(seed)
        rows = []
        for _ in range(length):
            point = {name: rng.uniform(-1., 1.) for name in self._controls}
            value = predict_action(self._truth, {"kind": "response", "point": point}, self.public_spec())
            rows.append(Observation(point, value))
        return rows

    def blind_evaluate(self, program, *, length=256, seed=1000003):
        if isinstance(program, dict):
            program = Program.from_dict(program)
        rows = self.evaluation_rows(length=length, seed=seed)
        predictions = [(program.predict(row) if isinstance(program, Program) else program(row.features))
                       for row in rows]
        loss = sum((p - row.target) ** 2 for p, row in zip(predictions, rows)) / len(rows)
        equivalent = None
        if isinstance(program, Program):
            # This exact finite-language classification is separate from held-out
            # accuracy. Arbitrary submitted code is not certified by a point test.
            expanded = program.expanded()
            spec = self.public_spec()
            actions = [a for a in self.available_actions() if a["kind"] == "response"]
            for candidate in public_candidates(spec):
                if expanded == public_program(candidate, spec):
                    equivalent = all(abs(predict_action(candidate, a, spec) -
                                         predict_action(self._truth, a, spec)) <= TOLERANCE
                                     for a in actions)
                    break
        return {"mse": loss, "independent_mse": loss, "correct": loss <= TOLERANCE,
                "structural_equivalence": equivalent, "rows": len(rows)}


class OnlineInquiryResearcher:
    """Shared finite hypothesis search with optional live support dependencies."""

    def __init__(self, *, mode="recursive", state=None, observation_budget=24,
                 seed=71, compute_budget=1000000):
        if mode not in {"recursive", "flat", "frozen_method"}:
            raise ValueError("mode must be recursive, flat or frozen_method")
        state = state or {}
        self.mode, self.observation_budget, self.compute_budget = mode, observation_budget, compute_budget
        self.seed = seed
        self.library = deepcopy(state.get("library", []))
        self.methods = deepcopy(state.get("methods", []))
        self.tasks = state.get("tasks", 0)

    def to_dict(self):
        return {"library": deepcopy(self.library), "methods": deepcopy(self.methods),
                "tasks": self.tasks, "seed": self.seed}

    def run(self, lab):
        spec = lab.public_spec()
        actions = lab.available_actions()
        candidates = public_candidates(spec)
        events, nodes, evidence, revisions = [], [], [], []
        compute = 0
        used_method_steps = set()
        active_method = self.methods[-1] if self.methods else None
        response_actions = [a for a in actions if a["kind"] == "response"]
        coordinate_axes = ("driver", "orientation")
        calibration_axes = ("calibration_sign", "calibration_offset")
        # Full response signatures identify observational equivalence in this
        # finite language (e.g., sign does not distinguish quadratic functions).
        response_keys = {tuple(c.values()): tuple(predict_action(c, a, spec) for a in response_actions)
                         for c in candidates}
        compute += len(candidates) * len(response_actions)

        def emit(kind, **details):
            record = {"event_id": len(events), "event": kind,
                      "observations_before": len(evidence), **details}
            events.append(record)
            return record["event_id"]

        def key(c, goal):
            if goal == "response_law":
                return response_keys[tuple(c.values())]
            return tuple(c[a] for a in (coordinate_axes if goal == "coordinate_mapping" else calibration_axes))

        def values(goal):
            return {key(c, goal) for c in candidates}

        def open_question(goal, parent=None):
            node_id = f"q{len(nodes)}"
            event_id = emit("question_opened", question_id=node_id, parent_id=parent,
                            goal=goal, competing_answers=len(values(goal)))
            node = {"question_id": node_id, "goal": goal, "parent_id": parent,
                    "opened_at": event_id, "opened_before_observation": len(evidence),
                    "status": "unresolved", "resolved_at": None, "answer": None}
            nodes.append(node)
            return node

        def entropy(counts):
            total = sum(counts)
            return -sum(n / total * math.log(n / total) for n in counts if n)

        def action_gain(goal, action):
            nonlocal compute
            if compute + len(candidates) > self.compute_budget:
                return None
            goal_counts = {}
            for c in candidates:
                goal_counts[key(c, goal)] = goal_counts.get(key(c, goal), 0) + 1
            prior = entropy(list(goal_counts.values()))
            buckets = {}
            for c in candidates:
                outcome = round(predict_action(c, action, spec), 10)
                bucket = buckets.setdefault(outcome, {})
                k = key(c, goal)
                bucket[k] = bucket.get(k, 0) + 1
            compute += len(candidates)
            remaining = sum(sum(bucket.values()) / len(candidates) * entropy(list(bucket.values()))
                            for bucket in buckets.values())
            return prior - remaining

        def best_action(goal, allowed):
            ranked = []
            for a in allowed:
                gain = action_gain(goal, a)
                if gain is None:
                    break
                # Stable tie breaking is shared; no truth- or task-label lookup.
                ranked.append((gain, -len(ranked), a))
            return max(ranked, key=lambda x: x[:2])[2] if ranked else None

        def instantiate_recipe(recipe):
            action = deepcopy(recipe)
            action.pop("goal", None)
            if "point_values" in action:
                action["point"] = dict(zip(spec["controls"], action.pop("point_values")))
            return action

        def adaptive_method_action(node, allowed):
            best = best_action(node["goal"], allowed)
            if active_method is None or best is None:
                return best
            for index, recipe in enumerate(active_method["actions"]):
                if index in used_method_steps or recipe["goal"] != node["goal"]:
                    continue
                used_method_steps.add(index)
                previous = instantiate_recipe(recipe)
                old_gain = action_gain(node["goal"], previous)
                new_gain = action_gain(node["goal"], best)
                if old_gain is None or new_gain is None:
                    return None
                changed = new_gain > old_gain + TOLERANCE and self.mode != "frozen_method"
                result = best if changed else previous
                detail = {"method_id": active_method["method_id"], "step_index": index,
                          "question_id": node["question_id"], "goal": node["goal"],
                          "previous_action": previous, "selected_action": deepcopy(result),
                          "old_information_gain": old_gain, "new_information_gain": new_gain,
                          "consumes_answers": [n["question_id"] for n in nodes
                                               if n["status"] == "resolved"]}
                emit("method_rewritten" if changed else "method_step_reused", **detail)
                if changed:
                    revisions.append(deepcopy(detail))
                return result
            return best

        def acquire(action, node, reason):
            nonlocal candidates, compute
            if (action is None or len(evidence) >= self.observation_budget or
                    compute + len(candidates) > self.compute_budget):
                return False
            decision = emit("experiment_selected", question_id=node["question_id"],
                            goal=node["goal"], action=deepcopy(action), reason=reason,
                            candidate_count=len(candidates),
                            answers_known={n["question_id"]: n["answer"] for n in nodes
                                           if n["status"] == "resolved"})
            result = lab.execute(action)
            evidence.append(result)
            previous_count = len(candidates)
            compute += len(candidates)
            candidates = [c for c in candidates if abs(predict_action(c, action, spec) - result["value"]) <= TOLERANCE]
            emit("observation_received", question_id=node["question_id"],
                 decision_event=decision, observation=deepcopy(result),
                 candidates_before=previous_count, candidates_after=len(candidates))
            if not candidates:
                emit("language_mismatch", question_id=node["question_id"])
            return bool(candidates)

        def resolve(node):
            if node["status"] == "resolved":
                return True
            if len(values(node["goal"])) != 1:
                return False
            answer = next(iter(values(node["goal"])))
            if node["goal"] == "response_law":
                answer = {k: candidates[0][k] for k in ("mechanism", "driver", "orientation")}
            else:
                axes = coordinate_axes if node["goal"] == "coordinate_mapping" else calibration_axes
                answer = dict(zip(axes, answer))
            node.update(status="resolved", answer=answer)
            node["resolved_at"] = emit("question_resolved", question_id=node["question_id"], answer=answer)
            return True

        def recurse(node):
            if resolve(node):
                return True
            support = {"response_law": "coordinate_mapping",
                       "coordinate_mapping": "sensor_calibration"}.get(node["goal"])
            if support and len(values(support)) > 1:
                child = open_question(support, node["question_id"])
                emit("parent_blocked", question_id=node["question_id"], depends_on=child["question_id"])
                if not recurse(child):
                    return False
                emit("answer_returned", question_id=child["question_id"], parent_id=node["question_id"],
                     answer=deepcopy(child["answer"]))
                emit("parent_resumed", question_id=node["question_id"],
                     consumes_answer=child["question_id"], answer=deepcopy(child["answer"]))
            while candidates and len(evidence) < self.observation_budget:
                if resolve(node):
                    return True
                kind = {"sensor_calibration": "calibrate", "coordinate_mapping": "measure",
                        "response_law": "response"}[node["goal"]]
                choices = [a for a in actions if a["kind"] == kind]
                action = adaptive_method_action(node, choices)
                if not acquire(action, node, "discriminate_current_question"):
                    return False
            return resolve(node) if candidates else False

        root = open_question("response_law")
        memory_source = None
        replay_failed = False
        # Try a stored executable law in its previous binding before rebuilding.
        # All modes see this identical proposal, evidence and candidate update.
        if self.library:
            saved = self.library[-1]
            hypothesis = saved["hypothesis"]
            old = public_program(hypothesis, spec)
            trial = {"kind": "response", "point": dict(zip(spec["controls"], (1., 1.)))}
            emit("memory_proposed", source_id=saved["source_id"], hypothesis=deepcopy(hypothesis))
            if acquire(trial, root, "test_stored_hypothesis"):
                expected = old.predict(Observation(trial["point"], 0.))
                replay_failed = abs(expected - evidence[-1]["value"]) > TOLERANCE
                emit("memory_assessed", source_id=saved["source_id"], rejected=replay_failed,
                     prediction=expected, observed=evidence[-1]["value"])

        if active_method and self.mode in {"recursive", "frozen_method"}:
            emit("method_replay_started", method_id=active_method["method_id"],
                 rewriting_allowed=self.mode == "recursive")
        if self.mode in {"recursive", "frozen_method"}:
            recurse(root)
        else:
            while candidates and len(evidence) < self.observation_budget:
                if resolve(root):
                    break
                if not acquire(best_action("response_law", actions), root, "flat_joint_information_gain"):
                    break
        if active_method and self.mode in {"recursive", "frozen_method"}:
            emit("method_replay_finished", resolved=bool(candidates) and len(values("response_law")) == 1,
                 used_steps=sorted(used_method_steps), fallback_allowed=True)

        confirmed = False
        program = None
        hypothesis = None
        direct_call = False
        if candidates and resolve(root):
            hypothesis = {k: candidates[0][k] for k in ("mechanism", "driver", "orientation")}
            program = public_program(hypothesis, spec)
            for source in reversed(self.library):
                if source["hypothesis"]["mechanism"] == hypothesis["mechanism"]:
                    definition = Definition.from_dict(source["definition"])
                    z = Program("feature", name=spec["controls"][hypothesis["driver"]])
                    if hypothesis["orientation"] == -1:
                        z = Program("mul", children=(Program("const", value=-1.), z))
                    program = Program("call", name=definition.definition_id, definition=definition, children=(z,))
                    memory_source, direct_call = source["source_id"], True
                    emit("stored_hypothesis_called", source_id=memory_source,
                         binding=deepcopy(hypothesis), call_name=definition.definition_id)
                    break
            emit("hypothesis_frozen", hypothesis=deepcopy(hypothesis), program=program.to_dict())
            rng = Random(self.seed + self.tasks * 101)
            losses = []
            for _ in range(3):
                action = {"kind": "response", "point": {name: rng.uniform(-1., 1.) for name in spec["controls"]}}
                prediction = program.predict(Observation(action["point"], 0.))
                if not acquire(action, root, "confirm_frozen_hypothesis"):
                    break
                losses.append((prediction - evidence[-1]["value"]) ** 2)
            confirmed = len(losses) == 3 and max(losses) <= TOLERANCE
            emit("fresh_confirmation", passed=confirmed, rows=len(losses), losses=losses)

        if confirmed and not direct_call:
            source_id = f"inquiry_concept_{len(self.library) + 1}"
            definition = Definition(source_id, ("z",),
                                    _law_program(hypothesis["mechanism"], Program("arg", name="z")), source_id)
            self.library.append({"source_id": source_id, "hypothesis": deepcopy(hypothesis),
                                 "definition": definition.to_dict()})
            emit("hypothesis_retained", source_id=source_id)
        if confirmed and self.mode != "frozen_method":
            recipe = []
            for event in events:
                if event["event"] != "experiment_selected" or event["reason"] in {
                        "confirm_frozen_hypothesis", "test_stored_hypothesis"}:
                    continue
                action = deepcopy(event["action"])
                action["goal"] = event["goal"]
                if "point" in action:
                    action["point_values"] = [action["point"][name] for name in spec["controls"]]
                    action["point"] = {}
                recipe.append(action)
            if recipe:
                method_id = f"inquiry_method_{len(self.methods) + 1}"
                change = {"method_id": method_id, "source_method": self.methods[-1]["method_id"] if self.methods else None,
                          "actions": recipe, "support_goals": [n["goal"] for n in nodes if n["parent_id"]]}
                self.methods.append(change)
                emit("method_retained", method_id=method_id, prior_rejected=replay_failed)
        self.tasks += 1
        return {"program": program.to_dict() if program else None, "hypothesis": hypothesis,
                "confirmed": confirmed, "direct_call": direct_call,
                "used_sources": [memory_source] if memory_source else [],
                "mode": self.mode, "events": events, "nodes": nodes, "observations": evidence,
                "online_subquestions": sum(n["parent_id"] is not None for n in nodes),
                "resolved_subquestions": sum(n["parent_id"] is not None and n["status"] == "resolved" for n in nodes),
                "method_revisions": revisions, "remaining_candidates": len(candidates),
                "costs": {"observation_cost": len(evidence), "measurement_cost": 0, "compute_cost": compute},
                "resource_limits": {"observations": self.observation_budget, "compute": self.compute_budget}}
