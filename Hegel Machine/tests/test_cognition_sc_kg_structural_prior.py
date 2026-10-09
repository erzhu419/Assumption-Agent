"""Optional integration with the archived SC-KG runtime; no simulator calls."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def test_paid_prefix_prior_survives_native_gp_construction_and_rebuild():
    root = Path(__file__).resolve().parents[1]
    historical = root / "artifacts/sc_kg_structural_prior_comparison_20261007/kg_source/SC-OLH-KG"
    if not (historical / "representation/meta_prior.py").is_file():
        pytest.skip("optional archived SC-KG source is unavailable")
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    # A fresh process prevents other optional SC-KG tests' imported checkout
    # from silently replacing the historical runtime under examination.
    program = textwrap.dedent("""
        import sys
        from copy import deepcopy
        from unittest.mock import patch
        import numpy as np

        sys.path.insert(0, sys.argv[1])
        from algorithms.single_olhkg import SingleOLHKGAlgorithm, SingleOLHKGConfig
        from hegel_machine.cognition.sc_kg_structural_prior import attach_structural_prior

        class PublicProblem:
            d, L, alpha, tau, sigma_level = 3, 100, .05, 0., .03
            problem_name = "paid_prefix_integration"

            def normalize(self, point):
                return np.asarray(point, dtype=float) / self.L

            def int_bounds(self):
                return np.zeros(self.d, dtype=int), np.full(self.d, self.L, dtype=int)

            def forbidden(self, *args, **kwargs):
                raise AssertionError("fitting paid data must not query simulation or truth")

            simulate = true_outputs = true_objectives = true_sigma = forbidden

        rng = np.random.default_rng(7329)
        points = rng.integers(0, 101, size=(32, 3))
        points[-1] = points[0]  # Preserve two distinct paid draws at one control.
        records = []
        for index, point in enumerate(points):
            z = point / 100.
            records.append({"point": point.tolist(), "output": [
                float(.35 + np.sum((z - [.2, .7, .7]) ** 2)),
                float((z[0] - .4) ** 2 + .3 * z[1] - .2 + index * .0001)]})
        original = deepcopy(records)
        problem = PublicProblem()
        attach_structural_prior(problem, records, seed=41)
        assert records == original
        prior = problem._structural_prior_adapter.meta_prior
        assert len(prior.source_records_) == 32
        assert [list(row.x) for row in prior.source_records_] == [row["point"] for row in records]
        np.testing.assert_array_equal([row.y for row in prior.source_records_],
                                      [row["output"] for row in records])
        assert prior.source_records_[0].x == prior.source_records_[-1].x
        assert prior.source_records_[0].y[1] != prior.source_records_[-1].y[1]

        # These are fitted spectral/group/shrinkage objects, not just profile flags.
        spectral = prior.stage1_spectral_basis.diagnostics()
        assert spectral["status"] == "fit" and spectral["low_frequency_prior"]
        assert spectral["orthogonalization"] == "symmetric"
        assert spectral["max_offdiag_gram"] < 1e-5
        assert prior.spectral_frequency_bank
        assert prior.spectral_additive_bank.feature_dim > 0
        assert 1 in prior.spectral_coefficient_prior

        bases = [problem.gpr_basis_map(i) for i in range(2)]
        selected = [basis.selected_basis for basis in bases]
        config = SingleOLHKGConfig(N=32, n0=32, use_state_coupling=False,
            use_state_basis=False, lambda_coupling=0., variance_mode="pooled")
        algorithm = SingleOLHKGAlgorithm(problem, config)
        assert all(model.basis_map is basis for model, basis in zip(algorithm.gpr, bases))
        assert [basis.selected_basis for basis in bases] == selected
        for row in records:
            point, output = tuple(row["point"]), np.asarray(row["output"])
            algorithm.history.append((point, output))
            algorithm.observations.setdefault(point, []).append(output)
        for index, basis in enumerate(bases):
            with patch.object(basis, "apply_coefficient_prior", wraps=basis.apply_coefficient_prior) as consume:
                algorithm._rebuild_gpr_from_history(index, replay_sequential=False)
                assert consume.call_count == 1
            assert np.isfinite(algorithm.gpr[index].posterior_mean((35, 55, 75)))
        assert len(algorithm.history) == 32
        assert len(algorithm.observations[tuple(points[0])]) == 2
    """)
    env = {**os.environ, "PYTHONPATH": str(root / "src"),
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    result = subprocess.run([sys.executable, "-c", program, str(historical)],
                            cwd=root, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
