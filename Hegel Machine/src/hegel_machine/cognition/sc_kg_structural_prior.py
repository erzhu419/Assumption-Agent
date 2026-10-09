"""Attach KG_op's existing four-property prior to a paid-data comparison.

The caller imports the historical KG_op runtime before using this module. Only
data acquisition is adapted: source fitting receives the shared task prefix.
The original prior, pilot gate and coefficient updates remain KG_op operations.
"""

from collections import defaultdict
from copy import deepcopy
from time import perf_counter


def attach_structural_prior(problem, records, *, seed):
    """Install the historical full-profile GP bases, retaining the optimizer.

    Records are current-task observations, not an independent source archive.
    Fitting and pilot selection reuse them without issuing simulator calls.
    """
    import numpy as np
    from performance.structural_ablation import STRUCTURAL_PRIOR_PROFILES
    from representation.meta_prior import (
        LearnedMetaPrior, MetaPriorProblemAdapter, SourceRecord,
    )

    class PaidPrefixPrior(LearnedMetaPrior):
        def _record_source_data(self, source_problems, n_records_per_domain, rng,
                                n_records_by_problem=None):
            return [SourceRecord(
                domain="current_task_paid_prefix",
                x=tuple(row["point"]),
                y=np.asarray(row["output"], dtype=float),
                descriptor=self.descriptor(problem, row["point"]),
                profile=np.asarray(problem.normalize(row["point"]), dtype=float),
                tau=float(problem.tau), alpha=float(problem.alpha),
                sigma_level=float(problem.sigma_level),
                constraint_sigma=float(problem.sigma_level),
                origin="random", replicate_count=1,
                replicates=np.asarray([row["output"]], dtype=float),
            ) for row in records]

    profile = STRUCTURAL_PRIOR_PROFILES["full"]
    options = {key.removeprefix("meta_"): value for key, value in profile.items()
               if key.startswith("meta_spectral_")}
    started = perf_counter()
    prior = PaidPrefixPrior(component_stage="spectral", seed=seed,
                            source_observation_mode="nominal", **options)
    prior.fit_from_source_problems([("current_task_paid_prefix", problem)],
                                  n_records_per_domain=len(records))
    source_fit_seconds = perf_counter() - started
    adapter = MetaPriorProblemAdapter(problem, prior)
    observations = defaultdict(list)
    for row in records:
        observations[tuple(row["point"])].append(np.asarray(row["output"], dtype=float))
    started = perf_counter()
    # The existing paid-data loader bypasses acquisition in _fit_initial_belief.
    # Preserve the native basis-fitting step before its GP reconstruction.
    bases = [adapter.gpr_basis_map(index) for index in range(2)]
    for index, basis in enumerate(bases):
        basis.fit_from_observations(observations, output_index=index)
    problem.prefer_direct_gpr_basis = True
    problem.gpr_basis_map = adapter.gpr_basis_map
    problem._structural_prior_adapter = adapter
    problem._structural_prior_metadata = {
        "profile": "full",
        "active_components": profile["structural_prior_active_components"],
        "prior_options": options,
        "data_role": "current_task_paid_prefix",
        "paid_records_reused": len(records),
        "new_simulator_calls": 0,
        "source_fit_seconds": source_fit_seconds,
        "initial_basis_fit_seconds": perf_counter() - started,
        "scope": "Four-property GP prior fitted on the common task prefix; original gate and posterior adaptation retained; no separate source archive, prior proposals or HVD changes.",
    }
    return structural_prior_diagnostics(problem)


def structural_prior_diagnostics(problem):
    """Record which of the enabled historical structures the pilot selected."""
    adapter = problem._structural_prior_adapter
    return deepcopy({**problem._structural_prior_metadata,
                     "basis_diagnostics": adapter.meta_basis_diagnostics()})
