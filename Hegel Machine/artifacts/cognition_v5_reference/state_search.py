"""Learn finite-output transition rules from observed controls and past outputs.

No task family or environment interface is consulted. A two-valued response
suggests a discrete observed state, whose transition can depend on a current
coordinate differently in each previous state. Competing reset interpretations,
state lags, feature choices and threshold directions remain executable ASTs.
"""

from dataclasses import dataclass
from itertools import product

from .language import Program
from .models import Candidate


@dataclass(frozen=True)
class _Rule:
    program: Program
    errors: int
    feature: str | None = None
    direction: str = "constant"
    alternatives: tuple[Program, ...] = ()


def _threshold(name, direction, boundary, levels):
    condition = Program(direction, children=(Program("feature", name=name),
                                             Program("const", value=boundary)))
    if levels == (0.0, 1.0):
        return condition
    return Program("if", children=(condition, Program("const", value=levels[1]),
                                    Program("const", value=levels[0])))


def _fit_rules(rows, indices, names, levels, budget):
    """Fit both stump directions with one sorted cumulative-count pass.

    Each feature fit costs its number of rows. Comparing different cuts uses
    cached class counts, so it does not reevaluate a program at every threshold.
    """
    high = sum(rows[index].target == levels[1] for index in indices)
    rules = [_Rule(Program("const", value=levels[0]), high),
             _Rule(Program("const", value=levels[1]), len(indices) - high)]
    for name in names:
        if not indices or not budget.spend(len(indices)):
            break
        ordered = sorted((rows[index].features[name], rows[index].target == levels[1]) for index in indices)
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
            middle = _threshold(name, direction, (left + right) / 2, levels)
            alternatives = tuple(_threshold(name, direction, left + fraction * (right - left), levels)
                                 for fraction in (0.25, 0.75))
            rules.append(_Rule(middle, rank[0], name, direction, alternatives))
    return sorted(rules, key=lambda rule: (rule.errors, rule.program.size))


def state_candidates(observations, *, budget, max_history=1, max_candidates=12):
    """Return stateful and memoryless explanations learned from observations.

    The default searches lag one; larger ``max_history`` compares longer lags
    under the same budget. About 600--900 candidate-row units usually cover a
    lag-one search with two observed controls and up to 96 rows. Feature/group
    fits, state partitions and target discovery are charged by row; composing
    previously scored branches costs three cached-error comparisons.
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

    for lag in range(1, max_history + 1):
        if not budget.spend(len(rows)):
            break
        mature = [index for index, row in enumerate(rows) if len(row.history) >= lag]
        if not mature or any("__target__" not in rows[index].history[-lag] for index in mature):
            continue
        initial = [index for index, row in enumerate(rows) if len(row.history) < lag]
        midpoint = (levels[0] + levels[1]) / 2
        off_indices = [index for index in mature if rows[index].history[-lag]["__target__"] < midpoint]
        on_indices = [index for index in mature if rows[index].history[-lag]["__target__"] >= midpoint]
        off_rules, on_rules, initial_rules = fitted(off_indices), fitted(on_indices), fitted(initial)

        # Independent startup behavior competes with either observed state as
        # the reset condition. Pooling reset rows can sharpen transition cuts
        # without assuming which output value is the initial state.
        families = [(off_rules, on_rules, initial_rules, "independent")]
        for reset in ("off", "on"):
            if initial:
                pooled = fitted([*(off_indices if reset == "off" else on_indices), *initial])
                families.append((pooled if reset == "off" else off_rules,
                                 pooled if reset == "on" else on_rules, None, reset))
        for off_options, on_options, startup_options, reset in families:
            variants = product(off_options[:2], on_options[:2], startup_options[:2] if startup_options else (None,))
            for off, on, separate_start in variants:
                startup = separate_start or (off if reset == "off" else on)
                errors = off.errors + on.errors + (startup.errors if separate_start is not None else 0)
                reason = f"Learn lag-{lag} state-conditioned thresholds with {reset} reset behavior"
                record(transition(lag, on.program, off.program, startup.program), errors, reason, True)
            # Preserve ambiguity inside observed transition intervals. These
            # alternatives have the same fitted labels, but new queries between
            # the adjacent controls can distinguish their predictions.
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

    scored.sort(key=lambda item: (item[0], item[1]))
    # Keep a memoryless competitor even when several equally fitting transition
    # boundaries occupy the leading positions. Neither family is forced active.
    selected = scored[:max_candidates]
    memoryless = next((item for item in scored if not item[4]), None)
    if memoryless is not None and selected and all(item[4] for item in selected):
        selected[-1] = memoryless
    return [Candidate(f"state:{index}:{budget.spent}", program, "composition" if stateful else "relation", reason)
            for index, (_, _, program, reason, stateful) in enumerate(selected)]
