"""The controller's evidence and prospective support decision after ranking."""

from dataclasses import dataclass
from math import sqrt
from statistics import mean, pstdev

from .evaluation import point_loss, predict
from .selection import AssessedCandidate


@dataclass(frozen=True)
class CandidateComparison:
    best: AssessedCandidate
    current: AssessedCandidate
    supported_candidates: list[AssessedCandidate]
    adopt: bool
    support: bool
    counterexample_gain: float | None
    fresh_gain: float | None
    evidence_gain: float | None
    margin: float
    support_unit: str
    known_response_gain: float | None
    cumulative_prediction_support: bool


def compare_candidate_adoption(assessed, incumbent, observations, reference,
                               fresh_reference, *, fallback, spec, config):
    """Reuse the original controller decision on already acquired comparisons.

    Ranking, its budget, and the prior freezing of candidates stay with the
    caller. ``reference`` excludes the fresh rows; prospective ranking uses
    their concatenation as reference and also passes fresh rows separately.
    The controller's later concept-merge execution-cost comparison remains at
    its call site because it requires the search budget and library state.
    """
    if not fresh_reference:
        raise ValueError('candidate adoption requires an acquired fresh comparison')
    best = assessed[0]
    current = next((row for row in assessed if row.candidate.program == incumbent.program), None)
    if current is None:
        raise ValueError('the incumbent must be included in the assessed candidates')
    prospective = config.selection_policy == 'prospective'
    cumulative_risk = bool(config.risk_learning and config.tail_fit and spec.decision_costs is not None)
    binary_state_support = spec.decision_costs is None and len({row.target for row in observations}) == 2
    supported_candidates = assessed
    if prospective and binary_state_support:
        known_count = len(observations) + len(fresh_reference)
        current_known = current.evidence_loss * len(observations) + current.fresh_loss * len(fresh_reference)
        supported_candidates = [row for row in assessed
            if row.evidence_loss * len(observations) + row.fresh_loss * len(fresh_reference)
            <= current_known + config.minimum_improvement * known_count]
        best = min(supported_candidates, key=lambda row: (row.score, row.candidate.program.size))
    counterexample_gain = None
    if prospective and spec.decision_costs is None:
        # A new random trajectory can miss the narrow region that an
        # earlier intervention already falsified. A coherent candidate
        # may resolve that contradiction, including by adding a needed
        # coordinate, while preserving every representative comparison.
        consistent = [row for row in supported_candidates
                      if row.evidence_loss <= config.minimum_improvement
                      and row.reference_loss <= current.reference_loss + config.minimum_improvement
                      and row.fresh_loss <= current.fresh_loss + config.minimum_improvement
                      and current.evidence_loss - row.evidence_loss > config.minimum_improvement
                      + config.complexity_weight * max(0, row.candidate.program.size - incumbent.program.size)]
        if consistent:
            best = min(consistent, key=lambda row: (row.score, row.candidate.program.size))
            counterexample_gain = current.evidence_loss - best.evidence_loss
    support = True
    margin, fresh_gain, evidence_gain = 0.0, None, None
    support_unit = "observation"
    known_response_gain = None
    cumulative_prediction_support = False
    if prospective and fresh_reference:
        # These differences are already represented in the paid fresh
        # evaluations. The dispersion is a decision heuristic, not a
        # hypothesis-test certificate from correlated trajectory rows.
        differences = [point_loss(predict(incumbent.program, r, fallback), r.target, spec.risk_quantile)
                       - point_loss(predict(best.candidate.program, r, fallback), r.target, spec.risk_quantile)
                       for r in ([*reference, *fresh_reference] if cumulative_risk else fresh_reference)]
        fresh_gain = mean(differences[-len(fresh_reference):])
        evidence_gain = mean(differences)
        margin = 0.5 * pstdev(differences) / sqrt(len(differences))
        if cumulative_risk:
            from .risk_search import trajectory_blocks
            blocks = trajectory_blocks([*reference, *fresh_reference])
            offset, trajectory_gains = 0, []
            for block in blocks:
                trajectory_gains.append(mean(differences[offset:offset+len(block)]))
                offset += len(block)
            # A single trajectory has no between-trajectory dispersion.
            # Keep the existing row screening until replication exists;
            # using the gain itself would make every positive gain pass.
            if len(trajectory_gains) > 1:
                margin = 0.5 * pstdev(trajectory_gains) / sqrt(len(trajectory_gains))
                support_unit = "trajectory"
        support = evidence_gain >= max(config.minimum_improvement, margin) or (
            abs(evidence_gain) <= config.minimum_improvement
            and best.candidate.program.size < incumbent.program.size
            and best.evidence_loss <= current.evidence_loss + config.minimum_improvement
            and best.reference_loss <= current.reference_loss + config.minimum_improvement)
        # New interventions may localize a boundary that a fresh
        # representative sample misses. An equally compact correction
        # can use that evidence when its new predictions do not worsen.
        if spec.decision_costs is None and fresh_gain >= -config.minimum_improvement:
            support = support or (best.evidence_loss + config.minimum_improvement < current.evidence_loss
                                  and best.candidate.program.size <= incumbent.program.size)
            complexity_increase = config.complexity_weight * max(
                0, best.candidate.program.size - incumbent.program.size)
            cumulative_prediction_support = bool(reference and
                best.evidence_loss + config.minimum_improvement + complexity_increase < current.evidence_loss and
                best.reference_loss + config.minimum_improvement + complexity_increase < current.reference_loss)
            support = support or cumulative_prediction_support
        if binary_state_support:
            # One newly corrected transition does not compensate for
            # erasing several acquired state counterexamples. Compare
            # the already-paid old and new responses on equal terms.
            count = len(observations) + len(fresh_reference)
            known_response_gain = ((current.evidence_loss - best.evidence_loss) * len(observations)
                + fresh_gain * len(fresh_reference)) / count
    adopt = (counterexample_gain is not None
             or (support and best.score + config.minimum_improvement < current.score))
    return CandidateComparison(best, current, supported_candidates, adopt, support,
        counterexample_gain, fresh_gain, evidence_gain, margin, support_unit,
        known_response_gain, cumulative_prediction_support)
