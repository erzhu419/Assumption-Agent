"""Pytest Suite for Hegel V18.1: Forensic Decision-Level Provenance Chains & Official Subscription Harness."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import pytest
import random

from hegel_machine.cognition.hipporag_wormhole_index import FlatEqualMetadataIndex, HippoRAGWormholeIndex
from hegel_machine.cognition.m5_live_gpt_and_provenance import (
    CodexSubscriptionClient,
    DecisionProvenanceLedger,
    _build_live_k15_forest_and_archive,
    build_m5_live_gpt_task_suite,
    run_single_live_gpt_m5_episode,
    run_traced_deterministic_m5_episode,
    verify_decision_provenance_chain,
)
from hegel_machine.cognition.m5_temporal_multiscale_and_base_subtree import (
    _build_seed_forest_and_priors,
    _generate_m5_benchmark_tasks,
    _run_genuine_solver_pipeline,
)


class _MockSubscriptionClient:
    """Deterministic mock client testing multi-turn `acquire_probe` + `author_and_confirm_python_model` + OOS abstain."""

    def __init__(self, model: str = "mock-chatgpt-sub") -> None:
        self.model = model
        self.provider_backend = "openai_chatgpt_subscription_codex_cli"
        self.turn = 0

    def call_tool_step(self, messages: list[dict], *, max_tokens: int = 2048) -> dict:
        del max_tokens
        self.turn += 1
        user_txt = messages[1]["content"]
        if "OOS_nonstationary_chirp" in user_txt or "L06_OutOfScope" in user_txt:
            return {
                "ok": True, "wall_ms": 5.0, "returned_model": self.model,
                "prompt_tokens": 1200, "cached_tokens": 400, "completion_tokens": 25,
                "tool_call_id": f"c_{self.turn}", "tool_name": "abstain_scoped_unknown",
                "tool_args": {
                    "rationale": "High-freq chirp",
                    "considered_alternatives": [{"option": "stationary_ode", "rejected_reason": "non_stationary_oscillation"}],
                },
                "assistant_message_for_history": {"role": "assistant", "action_taken": {"action_type": "abstain_scoped_unknown"}},
            }
        if self.turn == 1:
            return {
                "ok": True, "wall_ms": 5.0, "returned_model": self.model,
                "prompt_tokens": 1500, "cached_tokens": 0, "completion_tokens": 40,
                "tool_call_id": "c_1", "tool_name": "acquire_probe",
                "tool_args": {
                    "probe_id": "open_branch_discriminating_probe",
                    "rationale": "Disambiguate open branch",
                    "considered_alternatives": [{"option": "nuisance_high_var_aux", "rejected_reason": "zero_target_voi"}],
                },
                "assistant_message_for_history": {"role": "assistant", "action_taken": {"action_type": "acquire_probe"}},
            }
        code = (
            "import math\n"
            "import numpy as np\n"
            "from scipy.optimize import least_squares\n"
            "PARAMS = None\n"
            "def fit(rows):\n"
            "    global PARAMS\n"
            "    u = np.array([r['u_now'] for r in rows])\n"
            "    s = np.array([r['s_grad'] for r in rows])\n"
            "    y = np.array([r['y'] for r in rows])\n"
            "    res = least_squares(lambda p: p[0]*(1 - p[1]*s/(1+s))*(u/(1.2+u)) + p[2]*np.sqrt(u) - y, [6.5, 0.70, -0.3], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))\n"
            "    PARAMS = res.x\n"
            "def predict(r):\n"
            "    u, s = r['u_now'], r['s_grad']\n"
            "    return float(PARAMS[0]*(1 - PARAMS[1]*s/(1+s))*(u/(1.2+u)) + PARAMS[2]*math.sqrt(max(0.0, u)))\n"
        )
        return {
            "ok": True, "wall_ms": 8.0, "returned_model": self.model,
            "prompt_tokens": 1800, "cached_tokens": 1000, "completion_tokens": 160,
            "tool_call_id": f"c_{self.turn}", "tool_name": "author_and_confirm_python_model",
            "tool_args": {
                "hypothesis_name": "siphon_sqrt_fit",
                "python_code": code,
                "rationale": "Fit 2-layer siphon",
                "considered_alternatives": [{"option": "quadratic_storage", "rejected_reason": "ignores_s_grad"}],
            },
            "assistant_message_for_history": {"role": "assistant", "action_taken": {"action_type": "author_and_confirm_python_model"}},
        }


def test_subscription_client_rejects_non_chatgpt_and_scrubs_ruoli() -> None:
    assert "RUOLI_BASE_URL" in CodexSubscriptionClient._SCRUBBED_ENV_KEYS
    assert "GPT5_BASE_URL" in CodexSubscriptionClient._SCRUBBED_ENV_KEYS
    assert "OPENAI_BASE_URL" in CodexSubscriptionClient._SCRUBBED_ENV_KEYS
    with pytest.raises(RuntimeError):
        CodexSubscriptionClient(codex_info={"available": True, "auth_mode": "custom_ruoli"}, model="gpt-6-luna")


def test_provenance_hash_chain_tamper_and_parent_dag_detection() -> None:
    led = DecisionProvenanceLedger(run_id="r1", task_id="t1", arm="hegel", model="gpt-6-luna")
    d0 = led.record_decision(
        decision_type="D0_ROOT", parent_decision_id=None,
        evidence_inputs={"obs": [1, 2]}, candidates_evaluated=[{"c": "a"}],
        chosen_option={"c": "a"}, rejection_reasons={}, rationale_summary="root",
        measurement_delta=5.0, compute_delta=2.0,
    )
    led.record_decision(
        decision_type="D1_CHILD", parent_decision_id=d0,
        evidence_inputs={"obs": [3]}, candidates_evaluated=[{"c": "b"}],
        chosen_option={"c": "b"}, rejection_reasons={}, rationale_summary="child",
        measurement_delta=3.0, compute_delta=4.0,
    )
    ok_res = verify_decision_provenance_chain(led.records)
    assert ok_res["valid"] is True
    assert ok_res["verified_measurement_cost"] == 8.0
    assert ok_res["verified_compute_cost"] == 6.0

    tampered = deepcopy(led.records)
    tampered[0]["rationale_summary"] = "altered"
    assert verify_decision_provenance_chain(tampered)["valid"] is False

    bad_parent = deepcopy(led.records)
    bad_parent[1]["parent_decision_id"] = "dec__nonexistent"
    assert verify_decision_provenance_chain(bad_parent)["valid"] is False


def test_traced_deterministic_parity_with_untraced_pipeline() -> None:
    forest, pb = _build_seed_forest_and_priors()
    h_idx = HippoRAGWormholeIndex(forest)
    f_idx = FlatEqualMetadataIndex(forest)
    tasks = _generate_m5_benchmark_tasks(101)

    for ds in tasks[:8]:
        r_untraced = _run_genuine_solver_pipeline(
            ds, "Hegel_M5_Full_TemporalSubtree_Forest",
            forest=forest, priors_bundle=pb, hippo_idx=h_idx, flat_idx=f_idx, rng=random.Random(12345),
        )
        r_traced, chain = run_traced_deterministic_m5_episode(
            ds, "Hegel_M5_Full_TemporalSubtree_Forest",
            forest=forest, priors_bundle=pb, hippo_idx=h_idx, flat_idx=f_idx, rng=random.Random(12345),
        )
        v_chk = verify_decision_provenance_chain(chain)
        assert v_chk["valid"] is True
        assert r_traced["status"] == r_untraced["status"]
        assert r_traced["strict_pass"] == r_untraced["strict_pass"]
        assert abs(r_traced["measurement_cost"] - r_untraced["measurement_cost"]) < 1e-6


def test_live_k15_subscription_harness_and_oos_abstention(tmp_path: Path) -> None:
    forest, pb, arch = _build_live_k15_forest_and_archive()
    assert len(forest.nodes) == 15
    h_idx = HippoRAGWormholeIndex(forest)
    f_idx = FlatEqualMetadataIndex(forest)
    suite = build_m5_live_gpt_task_suite(20261120)

    l04 = next(t for t in suite if t.task_id == "L04_TwoLayer_SharedSubtree_Siphon")
    ep_ok, chain_ok = run_single_live_gpt_m5_episode(
        l04, "Hegel_M5_LiveGPT", client=_MockSubscriptionClient("gpt-6-luna"), workspace_dir=tmp_path / "l04",
        forest=forest, priors_bundle=pb, hippo_idx=h_idx, flat_idx=f_idx, node_archive=arch,
    )
    assert ep_ok["strict_pass"] is True
    assert ep_ok["false_confirmation"] is False
    assert ep_ok["provider_backend"] == "openai_chatgpt_subscription_codex_cli"
    assert verify_decision_provenance_chain(chain_ok)["valid"] is True

    l06 = next(t for t in suite if t.task_id == "L06_OutOfScope_NonStationary_Chirp")
    ep_oos, chain_oos = run_single_live_gpt_m5_episode(
        l06, "Hegel_M5_LiveGPT", client=_MockSubscriptionClient("gpt-6.1-terra"), workspace_dir=tmp_path / "l06",
        forest=forest, priors_bundle=pb, hippo_idx=h_idx, flat_idx=f_idx, node_archive=arch,
    )
    assert ep_oos["status"] == "scoped_unknown"
    assert ep_oos["strict_pass"] is True
    assert verify_decision_provenance_chain(chain_oos)["valid"] is True
