"""Learn finite-output transition rules from observed controls and past outputs.

No task family or environment interface is consulted. A two-valued response
suggests a discrete observed state, whose transition can depend on a current
coordinate differently in each previous state. Competing reset interpretations,
state lags, feature choices and threshold directions remain executable ASTs.
"""

from dataclasses import dataclass
from itertools import product
from math import exp

from .language import Program
from .models import Candidate
from .synthesis import _solve


@dataclass(frozen=True)
class _Rule:
    program: Program
    errors: int
    feature: str | None = None
    direction: str = "constant"
    alternatives: tuple[Program, ...] = ()


def _state_groups(rows, lag, levels):
    mature = [index for index, row in enumerate(rows) if len(row.history) >= lag]
    if not mature or any("__target__" not in rows[index].history[-lag] for index in mature):
        return None
    initial = [index for index, row in enumerate(rows) if len(row.history) < lag]
    midpoint = sum(levels) / 2
    off = [index for index in mature if rows[index].history[-lag]["__target__"] < midpoint]
    on = [index for index in mature if rows[index].history[-lag]["__target__"] >= midpoint]
    return off, on, initial


def _basic_transition_work(rows, names, groups, fitted_keys):
    # Partition discovery is one row pass. Every noncached nonempty group
    # then fits all raw coordinates; at most 12 complete branch combinations
    # and interval alternatives cost three cached comparisons each.
    if groups is None:
        return len(rows)
    unseen = {tuple(group) for group in groups} - set(fitted_keys)
    return len(rows) + len(names) * sum(map(len, unseen)) + 36


def state_search_work(observations, max_history=1):
    """Fund complete raw-coordinate initialization and the first state lag.

    This is a sufficient executable quote from acquired values and history,
    including independent startup/on/off fits. Joint coordinates, reset
    pooling and deeper lags use remaining search work. max_history=0 requests
    only memoryless fits; larger values still quote the first lag's minimum
    complete opportunity, rather than require all deeper explorations first.
    """
    rows = list(observations)
    levels = tuple(sorted({row.target for row in rows}))
    if len(levels) != 2:
        return 0
    names = sorted(set.intersection(*(set(row.features) for row in rows)))
    work = len(rows) * (1 + len(names)) + 2 + 2 * len(names)
    if max_history > 0:
        work += _basic_transition_work(rows, names, _state_groups(rows, 1, levels),
                                       {tuple(range(len(rows)))})
    return work


def _threshold(coordinate, direction, boundary, levels):
    condition = Program(direction, children=(coordinate,
                                             Program("const", value=boundary)))
    if levels == (0.0, 1.0):
        return condition
    return Program("if", children=(condition, Program("const", value=levels[1]),
                                    Program("const", value=levels[0])))


def _fit_rules(rows, indices, names, levels, budget, *, coordinate=None):
    """Fit both stump directions with one sorted cumulative-count pass.

    Each feature fit costs its number of rows. Comparing different cuts uses
    cached class counts, so it does not reevaluate a program at every threshold.
    """
    high = sum(rows[index].target == levels[1] for index in indices)
    rules = [_Rule(Program("const", value=levels[0]), high),
             _Rule(Program("const", value=levels[1]), len(indices) - high)]
    coordinates = [(Program("feature", name=name), name) for name in names]
    if coordinate is not None:
        coordinates = [(coordinate, None)]
    for expression, name in coordinates:
        if not indices or not budget.spend(len(indices)):
            break
        ordered = sorted((expression.predict(rows[index]), rows[index].target == levels[1]) for index in indices)
        left_high, best = 0, {}
        for offset, (value, label) in enumerate(ordered[:-1]):
            left_high += label
            right = ordered[offset + 1][0]
            if value == right:
                continue
            left_count = offset + 1
            ascending_errors = left_high + (len(indices) - left_count - (high - left_high))
            for direction, errors in (("ge", ascending_errors), ("lt", len(indices) - ascending_errors)):
                # Prefer a balanced supported cut when multiple empirical
                # minima exist; retain the interval around that cut below.
                rank = (errors, abs(2 * left_count - len(indices)))
                if direction not in best or rank < best[direction][0]:
                    best[direction] = (rank, value, right)
        for direction, (rank, left, right) in best.items():
            middle = _threshold(expression, direction, (left + right) / 2, levels)
            alternatives = tuple(_threshold(expression, direction, left + fraction * (right - left), levels)
                                 for fraction in (0.25, 0.75))
            rules.append(_Rule(middle, rank[0], name, direction, alternatives))
    return sorted(rules, key=lambda rule: (rule.errors, rule.program.size))


def startup_revision_candidates(programs, observations, *, budget, max_candidates=6):
    """Repair competing initial regimes while retaining their mature mechanisms.

    Existing state programs already contain an initial branch, so their
    minimum history depth is zero. They still need local revision when new
    starts contradict that branch. Reuse their executable current coordinates
    before trying raw inputs; the state guard and mature subtree stay intact.
    No mature-state fit or new experiment is needed for this local question.

    Candidates take turns fitting their first existing coordinate before any
    one candidate tries all raw alternatives. Grouping is shared by startup
    depth, old predictions by startup program, and fitted rules by coordinate.
    Each new group costs one acquired-row pass; each prediction or coordinate
    fit costs its startup rows; emitting a program costs one cached comparison.
    The ordinary controller still compares their complete predictions.
    """
    if max_candidates < 1:
        return []
    roots = []
    for program in programs:
        root = program.expanded() if program.op == "call" else program
        if root.op == "if" and root.children[0].op == "available" and root not in roots:
            roots.append(root)
    rows = list(observations)
    if not rows or not roots:
        return []
    groups, old_predictions, jobs = {}, {}, []
    for root in roots:
        guard, mature, startup = root.children
        if guard.lag not in groups:
            if not budget.spend(len(rows)):
                continue
            initial = [index for index, row in enumerate(rows) if len(row.history) < guard.lag]
            levels = tuple(sorted({row.target for row in rows}))
            names = set.intersection(*(set(rows[index].features) for index in initial)) if initial else set()
            groups[guard.lag] = initial, levels, names
        initial, levels, names = groups[guard.lag]
        if not initial or len(levels) != 2:
            continue
        key = guard.lag, startup
        if key not in old_predictions:
            if not budget.spend(len(initial)):
                continue
            old_predictions[key] = sum(startup.predict(rows[index]) != rows[index].target for index in initial)
        old_errors = old_predictions[key]
        if not old_errors:
            continue
        coordinates = []

        def add_coordinate(expression):
            if (expression.op != "const" and expression.history_depth == 0
                    and expression.required_features() <= names and expression not in coordinates):
                coordinates.append(expression)

        def collect(expression):
            if expression.op == "call":
                # A scalar call remains a coordinate; expand only to inspect
                # threshold structure hidden inside a callable mechanism.
                collect(expression.expanded())
                return
            if expression.op in {"ge", "gt", "le", "lt"}:
                for child in expression.children:
                    add_coordinate(child)
            for child in expression.children:
                collect(child)

        collect(startup)
        collect(mature)
        for name in sorted(names):
            add_coordinate(Program("feature", name=name))
        jobs.append({"root": root, "initial": initial, "levels": levels,
                     "old_errors": old_errors, "coordinates": coordinates, "scored": [], "seen": set()})

    fitted = {}
    reserve = min(max_candidates, len(jobs))
    for priority in range(max((len(job["coordinates"]) for job in jobs), default=0)):
        for job in jobs:
            if priority >= len(job["coordinates"]):
                continue
            coordinate = job["coordinates"][priority]
            guard, _, startup = job["root"].children
            key = guard.lag, coordinate
            if key not in fitted:
                if budget.remaining < len(job["initial"]) + reserve:
                    continue
                fitted[key] = _fit_rules(rows, job["initial"], (), job["levels"], budget, coordinate=coordinate)
            for rule in fitted[key]:
                if rule.errors >= job["old_errors"]:
                    continue
                for alternative in (rule.program, *rule.alternatives):
                    if alternative == startup or alternative in job["seen"]:
                        continue
                    job["seen"].add(alternative)
                    job["scored"].append((rule.errors, priority, alternative.size, alternative))
    for job in jobs:
        job["scored"].sort(key=lambda item: item[:3])
    results, emitted = [], set()
    while any(job["scored"] for job in jobs) and len(results) < max_candidates:
        for job in jobs:
            guard, mature, _ = job["root"].children
            while job["scored"]:
                errors, _, _, replacement = job["scored"].pop(0)
                revised = Program("if", children=(guard, mature, replacement))
                if revised in emitted:
                    continue
                if len(results) >= max_candidates or not budget.spend(1):
                    return results
                emitted.add(revised)
                results.append(Candidate(f"startup_revision:{len(results)}", revised, "composition",
                    f"Retain the mature mechanism and reduce observed startup errors from {job['old_errors']} to {errors}"))
                break
    return results


def _joint_coordinate(rows, groups, names, levels, budget, *, reserve):
    """Infer a shared affine direction with independently fitted state offsets.

    A pooled classification fit estimates the direction from all state groups;
    its intercepts are discarded and empirical transition intervals are fitted
    afterwards. This lets rare entry and exit examples constrain the same
    coordinate without declaring them to have the same threshold. Each Newton
    sweep charges one candidate-row pass over a cached, scaled design, as do
    design construction and the later interval fits.
    """
    if len(names) < 2 or budget.remaining < reserve + 3 * len(rows):
        return None
    budget.spend(len(rows))
    centers = [sum(row.features[name] for row in rows) / len(rows) for name in names]
    scales = [max(abs(row.features[name] - center) for row in rows) for name, center in zip(names, centers)]
    varying = [(name, center, scale) for name, center, scale in zip(names, centers, scales) if scale > 1e-10]
    if len(varying) < 2:
        return None
    labels = sorted(set(groups))
    width = len(varying) + len(labels)
    design = [[(row.features[name] - center) / scale for name, center, scale in varying]
              + [float(group == label) for label in labels] for row, group in zip(rows, groups)]
    targets = [float(row.target == levels[1]) for row in rows]
    coefficients = [0.0] * width
    for _ in range(8):
        if budget.remaining < reserve + len(rows) or not budget.spend(len(rows)):
            break
        predictions = [1 / (1 + exp(-max(-30.0, min(30.0, sum(x * value for x, value in zip(row, coefficients))))))
                       for row in design]
        weights = [max(1e-8, p * (1 - p)) for p in predictions]
        hessian = [[sum(weight * row[left] * row[right] for weight, row in zip(weights, design))
                    for right in range(width)] for left in range(width)]
        gradient = [sum(row[index] * (target - predicted)
                        for row, target, predicted in zip(design, targets, predictions)) for index in range(width)]
        for index in range(width):
            penalty = 0.01 if index < len(varying) else 1e-6
            hessian[index][index] += penalty
            gradient[index] -= penalty * coefficients[index]
        step = _solve(hessian, gradient)
        coefficients = [value + change for value, change in zip(coefficients, step)]
        if max(map(abs, step), default=0.0) < 1e-5:
            break
    slopes = [value / scale for value, (_, _, scale) in zip(coefficients, varying)]
    norm = max(map(abs, slopes), default=0.0)
    if norm < 1e-8:
        return None
    orientation = 1.0 if next(value for value in slopes if abs(value) > norm * 1e-7) > 0 else -1.0
    contributions = []
    for (name, _, _), slope in zip(varying, slopes):
        weight = orientation * slope / norm
        if abs(weight) < 1e-7:
            continue
        term = Program("feature", name=name)
        contributions.append(term if abs(weight - 1) < 1e-10 else
                             Program("mul", children=(Program("const", value=weight), term)))
    if len(contributions) < 2:
        return None
    coordinate = contributions[0]
    for contribution in contributions[1:]:
        coordinate = Program("add", children=(coordinate, contribution))
    return coordinate


def state_candidates(observations, *, budget, max_history=1, max_candidates=12,
                     joint_coordinates=True):
    """Return stateful and memoryless explanations learned from observations.

    The default searches lag one; larger ``max_history`` compares longer lags
    under the same budget. Raw coordinates remain inexpensive competitors.
    When their best explanation still makes observed errors, remaining budget
    learns a joint coordinate before extending to another state lag. Around
    2,200 candidate-row units cover this lag-one search with two controls and
    96 rows. Feature/group fits, state partitions and target discovery are
    charged by row; composing branches costs three cached-error comparisons.
    """
    rows = list(observations)
    if not rows or not budget.spend(len(rows)):
        return []
    levels = tuple(sorted({row.target for row in rows}))
    if len(levels) != 2:
        return []
    names = sorted(set.intersection(*(set(row.features) for row in rows)))
    all_indices = tuple(range(len(rows)))
    cache = {}

    def fitted(indices):
        key = tuple(sorted(indices))
        if key not in cache:
            cache[key] = _fit_rules(rows, key, names, levels, budget)
        return cache[key]

    scored, seen = [], set()

    def record(program, errors, reason, stateful):
        if program in seen or not budget.spend(3 if stateful else 1):
            return
        seen.add(program)
        scored.append((errors, program.size, program, reason, stateful))

    global_rules = fitted(all_indices)
    for rule in global_rules:
        record(rule.program, rule.errors, "Test a memoryless coordinate or constant explanation", False)

    def transition(lag, on, off, startup):
        state = Program("ge", children=(Program("lag", name="__target__", lag=lag),
                                         Program("const", value=(levels[0] + levels[1]) / 2)))
        mature = Program("if", children=(state, on, off))
        if on == off == startup:
            return startup
        return Program("if", children=(Program("available", lag=lag), mature, startup))

    def compose(lag, families, *, joint=False):
        for off_options, on_options, startup_options, reset in families:
            variants = product(off_options[:2], on_options[:2], startup_options[:2] if startup_options else (None,))
            for off, on, separate_start in variants:
                startup = separate_start or (off if reset == "off" else on)
                errors = off.errors + on.errors + (startup.errors if separate_start is not None else 0)
                coordinate_kind = "joint-coordinate " if joint else ""
                reason = f"Learn lag-{lag} state-conditioned {coordinate_kind}thresholds with {reset} reset behavior"
                record(transition(lag, on.program, off.program, startup.program), errors, reason, True)
            # Preserve ambiguity inside observed transition intervals. New
            # queries distinguish these equally supported threshold locations.
            off, on = off_options[0], on_options[0]
            startup = startup_options[0] if startup_options else (off if reset == "off" else on)
            errors = off.errors + on.errors + (startup.errors if startup_options else 0)
            for side, rule in (("off", off), ("on", on)):
                for alternative in rule.alternatives:
                    off_program = alternative if side == "off" else off.program
                    on_program = alternative if side == "on" else on.program
                    start_program = (off_program if reset == "off" else on_program if reset == "on" else startup.program)
                    record(transition(lag, on_program, off_program, start_program), errors,
                           f"Retain an unobserved lag-{lag} transition boundary interval", True)

    for lag in range(1, max_history + 1):
        groups = _state_groups(rows, lag, levels)
        if budget.remaining < _basic_transition_work(rows, names, groups, cache):
            break
        if not budget.spend(len(rows)):
            break
        if groups is None:
            continue
        off_indices, on_indices, initial = groups
        midpoint = (levels[0] + levels[1]) / 2
        off_rules, on_rules, initial_rules = fitted(off_indices), fitted(on_indices), fitted(initial)

        # First establish whether any raw coordinate can explain each group.
        # If those independently optimized branches still fail, pooling startup
        # rows or varying the cuts cannot remove the residual. Spend the next
        # available fit on a joint coordinate before refining the failed raw
        # family, so collecting more observations does not starve invention.
        compose(lag, [(off_rules, on_rules, initial_rules, "independent")])
        if joint_coordinates and scored and min(item[0] for item in scored) > 0:
            groups = ["startup" if len(row.history) < lag else
                      "on" if row.history[-lag]["__target__"] >= midpoint else "off" for row in rows]
            # Fund one complete independent-startup explanation first. Reset
            # pooling is a later refinement, not a prerequisite for inventing
            # an executable joint rule when the available allowance shrinks.
            reserve = len(rows) + 36
            coordinate = _joint_coordinate(rows, groups, names, levels, budget, reserve=reserve)
            if coordinate is not None:
                joint_cache = {}

                def joint_fitted(indices):
                    key = tuple(sorted(indices))
                    if key not in joint_cache:
                        joint_cache[key] = _fit_rules(rows, key, (), levels, budget, coordinate=coordinate)
                    return joint_cache[key]

                joint_off, joint_on, joint_initial = (joint_fitted(indices)
                                                     for indices in (off_indices, on_indices, initial))
                compose(lag, [(joint_off, joint_on, joint_initial, "independent")], joint=True)
                for reset in ("off", "on"):
                    if initial:
                        pooled_indices = [*(off_indices if reset == "off" else on_indices), *initial]
                        if budget.remaining >= len(pooled_indices) + 24:
                            pooled = joint_fitted(pooled_indices)
                            compose(lag, [(pooled if reset == "off" else joint_off,
                                           pooled if reset == "on" else joint_on, None, reset)], joint=True)
        # Independent startup behavior still competes with either observed
        # state as the reset condition. Pooling can sharpen supported cuts.
        raw_reset_families = []
        for reset in ("off", "on"):
            if initial:
                pooled = fitted([*(off_indices if reset == "off" else on_indices), *initial])
                raw_reset_families.append((pooled if reset == "off" else off_rules,
                                          pooled if reset == "on" else on_rules, None, reset))
        compose(lag, raw_reset_families)

    scored.sort(key=lambda item: (item[0], item[1]))
    # Keep a memoryless competitor even when several equally fitting transition
    # boundaries occupy the leading positions. Neither family is forced active.
    selected = scored[:max_candidates]
    memoryless = next((item for item in scored if not item[4]), None)
    if memoryless is not None and selected and all(item[4] for item in selected):
        selected[-1] = memoryless
    return [Candidate(f"state:{index}:{budget.spent}", program, "composition" if stateful else "relation", reason)
            for index, (_, _, program, reason, stateful) in enumerate(selected)]
