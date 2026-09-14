"""Revise parts of learned computations and undo empirical concept compression."""

from copy import deepcopy
from itertools import combinations
from statistics import mean

from .condition_learning import learn_condition
from .evaluation import initial_prediction, point_loss, predict
from .language import Program
from .models import Candidate, Observation, SearchBudget
from .sparse_search import sparse_candidates
from .synthesis import _fit_linear


def bind_answers(program, answers):
    """Substitute lexical argument leaves without changing nested definitions."""
    if program.op == "arg":
        return answers.get(program.name, program)
    return Program(program.op, program.value, program.name, program.lag,
                   tuple(bind_answers(child, answers) for child in program.children), program.definition)


def _opened(program):
    if program.op == "call":
        return bind_answers(program.definition.body, dict(zip(program.definition.arguments, program.children)))
    return program


def _replace(program, path, replacement):
    if not path:
        return replacement
    children = list(program.children)
    children[path[0]] = _replace(children[path[0]], path[1:], replacement)
    return Program(program.op, program.value, program.name, program.lag, tuple(children), program.definition)


def _replace_equal(program, previous, replacement):
    if program == previous:
        return replacement
    return Program(program.op, program.value, program.name, program.lag,
                   tuple(_replace_equal(child, previous, replacement) for child in program.children), program.definition)


def _mechanisms(program, path=()):
    if program.op == "if":
        return [item for index in (1, 2) for item in _mechanisms(program.children[index], path + (index,))]
    if program.op in {"add", "sub"}:
        return [item for index, child in enumerate(program.children)
                for item in _mechanisms(child, path + (index,))]
    if program.op == "mul" and any(child.op == "const" for child in program.children):
        index = next((index for index, child in enumerate(program.children) if child.op != "const"), None)
        return _mechanisms(program.children[index], path + (index,)) if index is not None else []
    return [(path, program)] if program.op != "const" else []


def _calls(program):
    result = [program] if program.op == "call" else []
    return result + [call for child in program.children for call in _calls(child)]


def _payload(row):
    return {"features": dict(row.features), "target": row.target,
            "history": [dict(point) for point in row.history]}


def _same_program(left, right):
    return left == right or left.expanded().simplified() == right.expanded().simplified()


def _assessed(program, assessments):
    return next((row for row in assessments if _same_program(row.candidate.program, program)), None)


def _comparison_losses(row):
    return {"evidence": row.evidence_loss, "reference": row.reference_loss, "fresh": row.fresh_loss}


def _scope(observations, program):
    terms = []
    for name, lag in sorted(program.observed_variables()):
        if lag and any(len(row.history) < lag for row in observations):
            continue
        if name == "__target__":
            continue
        coordinate = Program("lag", name=name, lag=lag) if lag else Program("feature", name=name)
        values = [coordinate.predict(row) for row in observations]
        low, high = min(values), max(values)
        lower = Program("ge", children=(coordinate, Program("const", value=low)))
        upper = Program("ge", children=(Program("const", value=high), coordinate))
        terms.append(Program("and", children=(lower, upper)))
    if not terms:
        return Program("const", value=1.)
    result = terms[0]
    for term in terms[1:]:
        result = Program("and", children=(result, term))
    return result


class KnowledgeEvolution:
    """The original computations remain available after a scoped merge proposal.

    Every returned program is a candidate for the ordinary controller. Fitting
    a repair or agreeing on acquired observations does not adopt a hypothesis.
    """

    def __init__(self):
        self.merges = []
        self.last_event = {}

    def to_dict(self):
        return {"merges": deepcopy(self.merges)}

    @classmethod
    def from_dict(cls, payload):
        value = cls()
        value.merges = deepcopy((payload or {}).get("merges", []))
        return value

    def local_revisions(self, program, spec, observations, budget, *, support_observations=(), limit=4):
        """Change one additive mechanism while retaining every other computation.

        A component's target is the observed response minus the unchanged
        surrounding program. Only that component's observables enter its fit.
        Successful old observations can supply a separating applicability
        condition; inseparable contradictions cannot silently discard that data.
        """
        start = budget.spent
        rows = list(observations)
        support = list(support_observations)
        all_rows = [*support, *rows]
        self.last_event = {"operation": "local_mechanism_revision", "tested": [], "compute_cost": 0}

        def finish(candidates=()):
            self.last_event["compute_cost"] = budget.spent - start
            return list(candidates)

        if len(rows) < 4 or limit < 1 or spec.risk_quantile != .5:
            return finish()
        opened = _opened(program)
        mechanisms = _mechanisms(opened)
        if len(mechanisms) < 2 or not budget.spend(len(all_rows)):
            return finish()
        fallback = initial_prediction(rows, spec.risk_quantile)
        before = [predict(program, row, fallback) for row in all_rows]
        positive = [row for row, value in zip(all_rows, before)
                    if abs(value - row.target) <= 1e-6 * max(1., abs(value), abs(row.target))]
        negative = [row for row, value in zip(rows, before[len(support):])
                    if abs(value - row.target) > 1e-6 * max(1., abs(value), abs(row.target))]
        if not negative:
            return finish()
        baseline = mean((row.target - value) ** 2 for row, value in zip(rows, before[len(support):]))
        condition = None
        if positive:
            local = SearchBudget(min(budget.remaining // 5, 500))
            result = learn_condition(positive, negative, local)
            budget.spend(local.spent)
            condition = Program.from_dict(result["condition"]) if result["condition"] else None
            self.last_event["condition_learning"] = result
        fit_rows = negative if condition is not None else rows
        proposals = []
        for index, (path, mechanism) in enumerate(mechanisms[:8]):
            applicable = [row for row in fit_rows if len(row.history) >= mechanism.minimum_history_depth]
            if len(applicable) < 4:
                continue
            # The enclosing addition/subtraction is linear in this one branch.
            # The two executions measure its offset and actual coefficient.
            if not budget.spend(2 * len(applicable)):
                break
            zero = _replace(opened, path, Program("const", value=0.))
            one = _replace(opened, path, Program("const", value=1.))
            residual_rows = []
            names = {name for name, _ in mechanism.observed_variables() if name != "__target__"}
            for row in applicable:
                offset = predict(zero, row, fallback)
                scale = predict(one, row, fallback) - offset
                if abs(scale) < 1e-10:
                    continue
                residual_rows.append(Observation(
                    {name: value for name, value in row.features.items() if name in names},
                    (row.target - offset) / scale,
                    tuple({name: value for name, value in point.items()
                           if name in names or name == "__target__"} for point in row.history)))
            if len(residual_rows) < 4:
                continue
            reserve = len(all_rows) * 3
            local = SearchBudget(max(0, (budget.remaining - reserve) // max(1, len(mechanisms[:8]) - index)))
            revisions = []
            if local.spend(2 * len(residual_rows)):
                # Expanding only the failed branch avoids depending on its
                # previous source law outside the newly retained condition.
                revisions.append(_fit_linear((mechanism.expanded(),), residual_rows))
            generated = sparse_candidates(residual_rows, budget=local,
                max_history=min(4, max(1, mechanism.history_depth)))
            revisions.extend(candidate.program for candidate in generated)
            budget.spend(local.spent)
            for revision in dict.fromkeys(revisions):
                if revision == mechanism:
                    continue
                branch = (Program("if", children=(condition, mechanism, revision))
                          if condition is not None else revision)
                candidate_program = _replace(opened, path, branch)
                if not budget.spend(len(all_rows)):
                    break
                after = [predict(candidate_program, row, fallback) for row in all_rows]
                measured = mean((row.target - value) ** 2 for row, value in zip(rows, after[len(support):]))
                retained = all((row.target - value) ** 2 <= (row.target - old) ** 2 + .002
                               for row, value, old in zip(support, after, before))
                self.last_event["tested"].append({"path": list(path), "mechanism": mechanism.describe(),
                    "replacement": branch.to_dict(), "loss_before": baseline, "loss_after": measured,
                    "old_support_preserved": retained})
                if retained and measured + .002 < baseline:
                    parents = tuple(dict.fromkeys(call.definition.source_concept_id for call in _calls(program)
                                                  if call.definition.source_concept_id))
                    candidate = Candidate(f"local_mechanism:{index}:{len(proposals)}", candidate_program,
                        "mechanism_repair", f"Replace component {path}; preserve the other acquired mechanisms", parents)
                    proposals.append((measured, candidate.program.size, candidate))
        proposals.sort(key=lambda item: item[:2])
        return finish(item[2] for item in proposals[:limit])

    def merge_equivalent(self, program, spec, observations, budget, *, limit=4):
        """Share distinct calls that currently make indistinguishable predictions.

        The rewrite is executable only within its measured coordinate range;
        elsewhere its original computation remains the fallback. Within-range
        counterexamples can later undo this empirical compression as well.
        """
        start = budget.spent
        rows = list(observations)
        self.last_event = {"operation": "merge_concepts", "tested": [], "compute_cost": 0}
        if not rows or limit < 1:
            return []
        opened = _opened(program)
        calls = list(dict.fromkeys(_calls(opened)))[:8]
        fallback = initial_prediction(rows, spec.risk_quantile)
        measured = {}
        for call in calls:
            if any(len(row.history) < call.minimum_history_depth or
                   not call.required_features().issubset(row.features) for row in rows):
                continue
            if not budget.spend(len(rows)):
                break
            measured[call] = [predict(call, row, fallback) for row in rows]
        proposals = []
        for left, right in combinations(measured, 2):
            if len(proposals) >= limit:
                break
            if any(abs(a - b) > 1e-6 * max(1., abs(a), abs(b))
                   for a, b in zip(measured[left], measured[right])):
                continue
            canonical, other = sorted((left, right), key=lambda node: (node.size, node.describe()))
            # Include scope-column construction plus both whole-program checks.
            scope_units = len(rows) * len(opened.observed_variables())
            if not budget.spend(scope_units + 2 * len(rows)):
                break
            condition = _scope(rows, opened)
            compressed = _replace_equal(opened, other, canonical)
            rewrite = Program("if", children=(condition, compressed, opened))
            if any(record["rewritten"] == rewrite.to_dict() for record in self.merges):
                continue
            before = [predict(opened, row, fallback) for row in rows]
            after = [predict(rewrite, row, fallback) for row in rows]
            original_loss = mean(point_loss(value, row.target, spec.risk_quantile) for row, value in zip(rows, before))
            revised_loss = mean(point_loss(value, row.target, spec.risk_quantile) for row, value in zip(rows, after))
            if revised_loss > original_loss + 1e-8:
                continue
            identifier = f"concept_merge:{len(self.merges) + 1}"
            record = {"merge_id": identifier, "task_id": spec.task_id,
                "original": opened.to_dict(), "compressed": compressed.to_dict(), "rewritten": rewrite.to_dict(),
                "condition": condition.to_dict(), "retained_call": canonical.to_dict(),
                "replaced_call": other.to_dict(), "observations": [_payload(row) for row in rows],
                "status": "proposed", "counterexamples": [],
                "loss_before": original_loss, "loss_after": revised_loss}
            self.merges.append(record)
            self.last_event["tested"].append({"merge_id": identifier, "loss_before": original_loss,
                                               "loss_after": revised_loss, "scope_rows": len(rows)})
            parents = tuple(dict.fromkeys(call.definition.source_concept_id for call in (left, right)
                                          if call.definition.source_concept_id))
            proposals.append(Candidate(identifier, rewrite, "concept_merge",
                "Share empirically equivalent calls inside the observed scope", parents))
        self.last_event["compute_cost"] = budget.spent - start
        return proposals

    def reopen_merges(self, program, spec, observations, budget):
        """Restore separate computations when new outcomes refute their merge."""
        start = budget.spent
        rows = list(observations)
        self.last_event = {"operation": "split_concepts", "tested": [], "compute_cost": 0}
        proposals = []
        fallback = initial_prediction(rows, spec.risk_quantile)
        opened = _opened(program)
        for record in self.merges:
            original = Program.from_dict(record["original"])
            rewritten = Program.from_dict(record["rewritten"])
            restored = _replace_equal(opened, rewritten, original)
            if restored == opened or not rows or not budget.spend(2 * len(rows)):
                continue
            before = [point_loss(predict(program, row, fallback), row.target, spec.risk_quantile) for row in rows]
            after = [point_loss(predict(restored, row, fallback), row.target, spec.risk_quantile) for row in rows]
            gain = mean(before) - mean(after)
            self.last_event["tested"].append({"merge_id": record["merge_id"], "gain": gain})
            if gain > .002:
                record["status"] = "counterexample_reopened"
                for row, old, new in zip(rows, before, after):
                    example = _payload(row)
                    if new + .002 < old and example not in record["counterexamples"]:
                        record["counterexamples"].append(example)
                parents = tuple(dict.fromkeys(call.definition.source_concept_id for call in _calls(restored)
                                              if call.definition.source_concept_id))
                proposals.append(Candidate(f"split:{record['merge_id']}", restored, "concept_split",
                    "New measurements distinguish computations compressed on earlier observations", parents))
        self.last_event["compute_cost"] = budget.spent - start
        return proposals

    def record_comparison(self, previous, selected, assessed, spec, fresh_rows):
        """Complete empirical merge/split decisions using already-paid scores.

        ``previous`` is the Program before comparison; ``selected`` is the
        actual resulting Candidate, including a retained parent after rollback.
        ``assessed`` contains the ordinary prospective AssessedCandidates. An
        equivalent program may have several origins, so operation labels never
        decide whether a split took place. ``fresh_rows`` is the number of new
        compared observations. This method performs no predictions.
        """
        if not fresh_rows:
            return []
        previous_opened, selected_opened = _opened(previous), _opened(selected.program)
        decisions = []
        for record in self.merges:
            original = Program.from_dict(record["original"])
            rewritten = Program.from_dict(record["rewritten"])
            restored = _replace_equal(previous_opened, rewritten, original)
            if restored != previous_opened:
                shared = _assessed(previous, assessed)
                separate = _assessed(restored, assessed)
                if shared is None or separate is None or shared.fresh_loss is None or separate.fresh_loss is None:
                    continue
                if _same_program(selected.program, restored):
                    operation = "split_adopted"
                    record["status"] = "split_adopted"
                    record["split_program"] = selected.program.to_dict()
                elif _same_program(selected.program, previous):
                    operation = "split_rejected" if record["status"] == "counterexample_reopened" else "merge_retained"
                    record["status"] = "adopted"
                else:
                    operation = "split_rejected"
                shared_program, separate_program = previous, restored
            else:
                # A merged candidate can also be selected through an equal
                # program origin, rather than the original proposal's ID.
                restored_selected = _replace_equal(selected_opened, rewritten, original)
                if restored_selected == selected_opened or not _same_program(restored_selected, previous):
                    continue
                shared = _assessed(selected.program, assessed)
                separate = _assessed(previous, assessed)
                if shared is None or separate is None or shared.fresh_loss is None or separate.fresh_loss is None:
                    continue
                operation = "merge_adopted"
                record["status"] = "adopted"
                shared_program, separate_program = selected.program, previous
            parents = list(dict.fromkeys(call.definition.source_concept_id
                for call in _calls(_opened(separate_program)) if call.definition.source_concept_id))
            event = {"operation": operation, "merge_id": record["merge_id"], "task_id": spec.task_id,
                "selected_candidate_id": selected.candidate_id, "selected_operation": selected.operation,
                "shared_loss": _comparison_losses(shared), "separate_loss": _comparison_losses(separate),
                "fresh_rows": fresh_rows, "source_concept_ids": parents,
                "shared_program": shared_program.to_dict(), "separate_program": separate_program.to_dict(),
                "selected_program": selected.program.to_dict()}
            record.setdefault("comparisons", []).append(deepcopy(event))
            decisions.append(event)
        return decisions

    def bind_concept(self, concept_id, program):
        """Link actually learned programs to their empirical merge history.

        KnowledgeLibrary owns executable scope revision and source dependencies.
        A restored computation depends on its component definitions, not on the
        invalid compression it replaced. Binding does not change those programs.
        """
        bindings = []
        for record in self.merges:
            if record["status"] == "split_adopted" and record.get("split_program") and _same_program(
                    program, Program.from_dict(record["split_program"])):
                role, key = "split", "split_concept_id"
            elif record["status"] == "adopted" and _same_program(program, Program.from_dict(record["rewritten"])):
                role, key = "merge", "merged_concept_id"
            else:
                continue
            if record.get(key) != concept_id:
                record[key] = concept_id
                bindings.append({"operation": "bind_evolution_knowledge", "merge_id": record["merge_id"],
                                 "role": role, "concept_id": concept_id})
        return bindings
