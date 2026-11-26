"""Hegel M9.4 Persistent Mechanism–Executable Binding & Hidden-Schedule Continual World (`S -> T -> G -> C -> R`).

Unifies `M8` (`PersistentMechanismGraph`, `TypedMechanism`, `TaskModel`, `M8VariationalPlanner`, `AncestorReweight`,
`LocalSplit`, `sentinel` zero-drift isolation) with `M9` (`L1/L2` air-gapped container modules, 4-tuple qualification,
3-valued `ObservablePreconditionGuard`, and `ScientificProbeReceipt`) via explicit `MechanismExecutableBinding` records
WITHOUT building a third parallel framework and WITHOUT requiring arbitrary numerical Python code to be differentiable!

Continual Hidden-Schedule Design (Neutral Task IDs — zero stage/mechanism hints in task IDs):
  - Chain Alpha (`Polymer Coupon Hereditary Creep + Shared Multi-Lab Load-Cell Calibration + Post-Yield Viscoplasticity`):
      * `alpha_s00__polymer_coupon_creep_baseline`      (`S`: Source discovery + initial `MechanismExecutableBinding` @ `param_v1`)
      * `alpha_ep01__preloaded_coupon_compliance_batch` (`T`: Pre-loaded initial memory + geometric `past_dt` + secant & crossing)
      * `alpha_ep02__facility_recalibrated_coupon_batch`(`G`: Global shared load-cell calibration shift `param_v1 -> param_v2`; `sentinel` drift `0.0`)
      * `alpha_ep03__high_load_structural_coupon_batch` (`C`: High-load overstress counterexample in `param_v2` epoch -> component-scoped `C1` split)
      * `alpha_ep04__standard_load_fresh_coupon_batch`  (`R`: Fresh undamaged sub-yield specimen in `param_v2` epoch -> guard rejects `C1`, routes `C0@param_v2`)

  - Chain Beta (`2D Multi-Shift Reaction-Transport + Shared Sensor-Bank Calibration + Non-Separable Cross-Pocket + Box-Simplex KKT`):
      * `beta_s00__tracer_transport_tomography_baseline`      (`S`: Separable Kronecker Sylvester + KKT certificate @ `param_v1`)
      * `beta_ep01__anisotropic_multibudget_survey_batch`     (`T`: Anisotropic reversed-advection grid + 3-budget Pareto KKT)
      * `beta_ep02__recalibrated_sensor_array_survey_batch`   (`G`: Global shared sensor-bank gain/offset shift `param_v1 -> param_v2`; `sentinel` drift `0.0`)
      * `beta_ep03__dual_tracer_reactive_survey_batch`        (`C`: Additive-separable Channel A + non-separable vortex Channel B in `param_v2` epoch)
      * `beta_ep04__single_tracer_background_survey_batch`    (`R`: Nonzero ADDITIVE background `V(x,y)=a(x)+b(y)` in `param_v2` epoch -> `F02` guard ratio `0.0` routes `C0@param_v2`)
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import m6_frontier_real_benchmarks as m6f
from .m7_autogenous_forest_pretraining import (
    CLEAN_SCIPY_SANDBOX_IMAGE,
    ObservablePreconditionGuard,
    evaluate_observable_precondition_guard,
)
from .m7_holdout_ood_blind_benchmark import HoldoutOODTask, run_holdout_private_verifier
from .m8_persistent_variational_planner import (
    M8VariationalPlanner,
    PersistentMechanismGraph,
    TypedMechanism,
    evaluate_committed_task_model,
)
from .m9_3_online_lifecycle_benchmark import _SHARED_BOX_SIMPLEX_KKT_CHECKER_PY

M94_BINDINGS_FOREST_KEY = "__m9_4_mechanism_executable_bindings__"


# ============================================================================
# 1. Explicit `MechanismExecutableBinding` Layer (`M8 Graph <-> M9 Executables`)
# ============================================================================


VALID_BINDING_QUALIFICATION_IDS: frozenset[str] = frozenset(
    {
        "hegel_l1_qual_4tuple_v2",
        "hegel_l1_qual_4tuple_v3",
        "hegel_l2_aufhebung_v2",
        "hegel_public_scientific_gate_v4",
        "unit_test_synthetic_fixture_v1",
    }
)
VALID_EXECUTABLE_KINDS: frozenset[str] = frozenset(
    {"raw_source", "compiled_L2", "primitive_L1", "bound_adapter"}
)


@dataclass
class MechanismExecutableBinding:
    """Explicit versioned binding between a `PersistentMechanismGraph` node/component and a verified container executable (`M9.4.1` `Gate B`)."""

    binding_id: str
    chain_id: str  # 'alpha' | 'beta'
    branch_tag: str  # 'C0_base' | 'C1_extension'
    mechanism_id: str
    mechanism_version: int
    forest_node_id: str
    component_id: str
    preserved_component_ids: list[str]
    violated_component_id: str | None
    module_stem: str
    module_digest: str
    function_name: str
    qualification_id: str
    parameter_mechanism_id: str
    parameter_version: int
    executable_kind: str = "raw_source"
    source_artifact_sha256: str = ""
    compiled_skill_sha256: str = ""
    adapter_code_sha256: str = ""
    parameter_epoch_sha256: str = ""
    contract_schema_sha256: str = ""
    environment_digest: str = ""
    input_role_map: dict[str, str] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    shapes: dict[str, str] = field(default_factory=dict)
    time_context: dict[str, Any] = field(default_factory=dict)
    precondition_guard: dict[str, Any] = field(default_factory=dict)
    output_schema_keys: list[str] = field(default_factory=list)
    evidence_receipt_id: str = ""
    parameterized_closure_code: str = ""
    parameterized_entrypoint: str = ""
    parameterized_closure_sha256: str = ""
    dependent_artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MechanismExecutableBinding:
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: deepcopy(v) for k, v in data.items() if k in allowed})


def verify_binding_identity_and_foreign_keys(
    binding: MechanismExecutableBinding,
    *,
    graph: PersistentMechanismGraph | None = None,
    require_closure_code: bool = False,
) -> None:
    """Fail-closed identity, digest, qualification, and foreign-key verifier (`M9.4.1` `Gate B` / `A05`)."""
    if binding.executable_kind not in VALID_EXECUTABLE_KINDS:
        raise ValueError(f"Invalid executable_kind `{binding.executable_kind}` on `{binding.binding_id}`")
    if binding.qualification_id not in VALID_BINDING_QUALIFICATION_IDS:
        raise ValueError(f"Unqualified or unrecognized qualification_id `{binding.qualification_id}` on `{binding.binding_id}`")
    if require_closure_code and (not binding.parameterized_closure_code.strip() or not binding.parameterized_entrypoint.strip()):
        raise ValueError(f"Missing executable closure code/entrypoint on `{binding.binding_id}`")
    if binding.parameterized_closure_code:
        actual_sha = hashlib.sha256(binding.parameterized_closure_code.encode("utf-8")).hexdigest()
        if binding.parameterized_closure_sha256 and binding.parameterized_closure_sha256 != actual_sha:
            raise ValueError(
                f"Declared closure SHA-256 mismatch on `{binding.binding_id}`: "
                f"declared={binding.parameterized_closure_sha256[:16]} != actual={actual_sha[:16]}"
            )
        if binding.source_artifact_sha256 and binding.source_artifact_sha256 != actual_sha:
            raise ValueError(f"Source artifact SHA-256 mismatch on `{binding.binding_id}`")
    if graph is not None:
        if binding.parameter_mechanism_id not in graph.mechanisms:
            raise ValueError(f"Missing parameter_mechanism_id `{binding.parameter_mechanism_id}` in PersistentMechanismGraph")
        if int(binding.parameter_version) != int(graph.mechanisms[binding.parameter_mechanism_id].version):
            raise ValueError(
                f"Stale/mismatched parameter_version on `{binding.binding_id}`: "
                f"binding={binding.parameter_version} != graph={graph.mechanisms[binding.parameter_mechanism_id].version}"
            )
        if binding.mechanism_id not in graph.mechanisms:
            raise ValueError(f"Dangling mechanism_id `{binding.mechanism_id}` not registered in PersistentMechanismGraph.mechanisms")


class M94BindingRegistry:
    """Persists `MechanismExecutableBinding`s directly inside `PersistentMechanismGraph.forest.questions`."""

    def __init__(self, graph: PersistentMechanismGraph) -> None:
        self.graph = graph
        self.bindings: dict[str, MechanismExecutableBinding] = {}
        self.parameter_epochs: dict[str, dict[str, Any]] = {}
        self.binding_events: list[dict[str, Any]] = []
        self._hydrate()

    def _ensure_typed_mechanism_registered(self, binding: MechanismExecutableBinding) -> None:
        if binding.mechanism_id not in self.graph.mechanisms:
            parent_mech = binding.parameter_mechanism_id if binding.parameter_mechanism_id in self.graph.mechanisms else None
            feat_nms = tuple(sorted(binding.input_role_map.keys())) if binding.input_role_map else (binding.component_id,)
            self.graph.register_mechanism(
                TypedMechanism(
                    mechanism_id=binding.mechanism_id,
                    parent_id=parent_mech,
                    role="bound_constitutive_executable",
                    feature_names=feat_nms,
                    version=int(binding.mechanism_version),
                    shared_weights={},
                    provenance_note=f"Bound executable `{binding.binding_id}` ({binding.executable_kind}) on `{binding.forest_node_id}`.",
                ),
                tree_id=f"tree_m94_chain_{binding.chain_id}",
            )

    def _hydrate(self) -> None:
        blob = self.graph.forest.questions.get(M94_BINDINGS_FOREST_KEY)
        if isinstance(blob, dict) and blob.get("schema") in {"hegel_m9_4_bindings_v1", "hegel_m9_4_1_bindings_v2"}:
            self.parameter_epochs = deepcopy(blob.get("parameter_epochs") or {})
            self.binding_events = list(blob.get("binding_events") or [])
            self.bindings = {}
            for k, v in (blob.get("bindings") or {}).items():
                b = MechanismExecutableBinding.from_dict(v)
                self._ensure_typed_mechanism_registered(b)
                verify_binding_identity_and_foreign_keys(b, graph=self.graph, require_closure_code=False)
                self.bindings[str(k)] = b

    def sync_to_graph(self) -> None:
        self.graph.forest.questions[M94_BINDINGS_FOREST_KEY] = {
            "schema": "hegel_m9_4_1_bindings_v2",
            "bindings": {k: b.to_dict() for k, b in sorted(self.bindings.items())},
            "parameter_epochs": deepcopy(self.parameter_epochs),
            "binding_events": deepcopy(self.binding_events),
        }
        self.graph.sync_authoritative_state_to_forest()

    def register_binding(self, binding: MechanismExecutableBinding, *, reason: str = "registered") -> None:
        if binding.parameterized_closure_code and not binding.parameterized_closure_sha256:
            binding.parameterized_closure_sha256 = hashlib.sha256(binding.parameterized_closure_code.encode("utf-8")).hexdigest()
        if binding.parameterized_closure_sha256 and not binding.source_artifact_sha256:
            binding.source_artifact_sha256 = binding.parameterized_closure_sha256
        if binding.module_digest and not binding.compiled_skill_sha256:
            binding.compiled_skill_sha256 = binding.module_digest
        if not binding.contract_schema_sha256 and binding.output_schema_keys:
            binding.contract_schema_sha256 = hashlib.sha256(json.dumps(sorted(binding.output_schema_keys)).encode("utf-8")).hexdigest()
        self._ensure_typed_mechanism_registered(binding)
        verify_binding_identity_and_foreign_keys(binding, graph=self.graph, require_closure_code=False)
        self.bindings[binding.binding_id] = binding
        self.binding_events.append(
            {
                "event": reason,
                "binding_id": binding.binding_id,
                "executable_kind": binding.executable_kind,
                "chain_id": binding.chain_id,
                "branch_tag": binding.branch_tag,
                "mechanism_id": binding.mechanism_id,
                "mechanism_version": binding.mechanism_version,
                "parameter_mechanism_id": binding.parameter_mechanism_id,
                "parameter_version": binding.parameter_version,
                "source_artifact_sha256": binding.source_artifact_sha256[:16],
                "compiled_skill_sha256": binding.compiled_skill_sha256[:16],
            }
        )
        self.sync_to_graph()

    def update_shared_parameter_epoch(
        self,
        *,
        parameter_mechanism_id: str,
        new_version: int,
        parameter_payload: dict[str, Any],
        affected_chain_id: str,
        evidence_note: str,
    ) -> list[str]:
        """Propagate an `AncestorReweight` shared-parameter update to all dependent `MechanismExecutableBinding`s on `affected_chain_id`."""
        self.parameter_epochs[parameter_mechanism_id] = {
            "parameter_mechanism_id": parameter_mechanism_id,
            "version": int(new_version),
            "payload": deepcopy(parameter_payload),
            "evidence_note": evidence_note,
        }
        updated_ids: list[str] = []
        for bid, b in self.bindings.items():
            if b.parameter_mechanism_id == parameter_mechanism_id and b.chain_id == affected_chain_id:
                b.parameter_version = int(new_version)
                updated_ids.append(bid)
        self.binding_events.append(
            {
                "event": "shared_parameter_epoch_updated",
                "parameter_mechanism_id": parameter_mechanism_id,
                "new_version": int(new_version),
                "updated_binding_ids": updated_ids,
                "evidence_note": evidence_note,
            }
        )
        self.sync_to_graph()
        return updated_ids


def initialize_continual_mechanism_graph_with_sentinel(
    forest: Any | None = None,
) -> tuple[PersistentMechanismGraph, M8VariationalPlanner, M94BindingRegistry]:
    """Initialize `PersistentMechanismGraph` with shared calibration roots for Chain Alpha, Chain Beta, AND an unrelated
    `sentinel_acoustic_calibration` root + committed `TaskModel` (`0.0` collateral drift sentinel!).
    """
    graph = PersistentMechanismGraph(forest=forest)
    planner = M8VariationalPlanner(graph, min_gain_nats=0.25, collateral_tolerance_nats=-0.05)

    # 1. Unrelated Sentinel Root + Task (`0.0` drift control across ALL Alpha/Beta `G` & `C` revisions!)
    rng_s = np.random.default_rng(94001)
    x_s_tr = rng_s.uniform(-1.5, 1.5, size=36)
    z_s_tr = rng_s.uniform(-1.0, 1.0, size=36)
    y_s_tr = 1.65 * x_s_tr - 0.45 * z_s_tr + rng_s.normal(scale=0.01, size=36)
    x_s_va = rng_s.uniform(-1.5, 1.5, size=20)
    z_s_va = rng_s.uniform(-1.0, 1.0, size=20)
    y_s_va = 1.65 * x_s_tr[:20] * 0.0 + 1.65 * x_s_va - 0.45 * z_s_va + rng_s.normal(scale=0.01, size=20)
    graph.register_mechanism(
        TypedMechanism(
            mechanism_id="sentinel_shared_acoustic_root",
            parent_id=None,
            role="shared_upstream_premise",
            feature_names=("freq_norm", "temp_norm"),
            shared_weights={"freq_norm": 1.65, "temp_norm": -0.45},
            provenance_note="Unrelated acoustic calibration sentinel root (must maintain 0.0 drift).",
        ),
        tree_id="tree_m94_sentinel",
    )
    planner.register_task(
        task_id="sentinel_acoustic_calibration",
        shared_upstream_id="sentinel_shared_acoustic_root",
        feature_names=("freq_norm", "temp_norm"),
        train_data={"x": {"freq_norm": x_s_tr, "temp_norm": z_s_tr}, "y": y_s_tr},
        val_data={"x": {"freq_norm": x_s_va, "temp_norm": z_s_va}, "y": y_s_va},
        initial_local_shrinkage=50.0,
    )

    # 2. Chain Alpha Shared Upstream Calibration Mechanism (`param_v1`: unit stress load-cell gain=1, bias=0 across 4 labs)
    alpha_cal_cols = ("cmd_l0", "cmd_l1", "cmd_l2", "cmd_l3", "ind_l0", "ind_l1", "ind_l2", "ind_l3")
    alpha_v1_weights = {
        "cmd_l0": 1.0, "cmd_l1": 1.0, "cmd_l2": 1.0, "cmd_l3": 1.0,
        "ind_l0": 0.0, "ind_l1": 0.0, "ind_l2": 0.0, "ind_l3": 0.0,
    }
    graph.register_mechanism(
        TypedMechanism(
            mechanism_id="alpha_shared_loadcell_calibration",
            parent_id=None,
            role="shared_upstream_premise",
            feature_names=alpha_cal_cols,
            version=1,
            shared_weights=dict(alpha_v1_weights),
            provenance_note="Chain Alpha shared 4-lab load-cell actuator-to-true-stress calibration.",
        ),
        tree_id="tree_m94_chain_alpha",
    )

    # 3. Chain Beta Shared Upstream Calibration Mechanism (`param_v1`: unit sensor-bank gain=1, offset=0 across 4 banks)
    beta_cal_cols = ("ref_b0", "ref_b1", "ref_b2", "ref_b3", "ind_b0", "ind_b1", "ind_b2", "ind_b3")
    beta_v1_weights = {
        "ref_b0": 1.0, "ref_b1": 1.0, "ref_b2": 1.0, "ref_b3": 1.0,
        "ind_b0": 0.0, "ind_b1": 0.0, "ind_b2": 0.0, "ind_b3": 0.0,
    }
    graph.register_mechanism(
        TypedMechanism(
            mechanism_id="beta_shared_sensor_bank_calibration",
            parent_id=None,
            role="shared_upstream_premise",
            feature_names=beta_cal_cols,
            version=1,
            shared_weights=dict(beta_v1_weights),
            provenance_note="Chain Beta shared 4-bank tracer sensor gain/offset calibration.",
        ),
        tree_id="tree_m94_chain_beta",
    )

    reg = M94BindingRegistry(graph)
    reg.update_shared_parameter_epoch(
        parameter_mechanism_id="alpha_shared_loadcell_calibration",
        new_version=1,
        parameter_payload={"stress_gains": [1.0, 1.0, 1.0, 1.0], "stress_biases": [0.0, 0.0, 0.0, 0.0]},
        affected_chain_id="alpha",
        evidence_note="initial_stage_S_nominal_calibration",
    )
    reg.update_shared_parameter_epoch(
        parameter_mechanism_id="beta_shared_sensor_bank_calibration",
        new_version=1,
        parameter_payload={"bank_gains": [1.0, 1.0, 1.0, 1.0], "bank_offsets": [0.0, 0.0, 0.0, 0.0]},
        affected_chain_id="beta",
        evidence_note="initial_stage_S_nominal_calibration",
    )
    return graph, planner, reg


def run_m8_shared_calibration_update_from_public_dataset(
    *,
    chain_id: str,
    episode_id: str,
    public_item: dict[str, Any],
    graph: PersistentMechanismGraph,
    planner: M8VariationalPlanner,
    registry: M94BindingRegistry,
) -> dict[str, Any]:
    """Execute genuine `M8VariationalPlanner` (`AncestorReweight` vs `NoChange`) on observable facility/sensor
    calibration coupons/sweeps, verifying `0.0` collateral drift on `sentinel_acoustic_calibration`!
    """
    sentinel_tm = graph.active_task_models["sentinel_acoustic_calibration"]
    sentinel_va = graph.task_datasets["sentinel_acoustic_calibration"]["val"]
    sentinel_before = np.asarray(evaluate_committed_task_model(sentinel_tm, graph, sentinel_va)["predictions"], dtype=float)

    if chain_id == "alpha":
        ds = public_item["dataset"]
        cal = ds["calibration_load_cells"]
        labs = np.asarray(cal["lab_id"], dtype=int).ravel()
        cmds = np.asarray(cal["cmd_stress"], dtype=float).ravel()
        trues = np.asarray(cal["true_stress"], dtype=float).ravel()
        cols = ("cmd_l0", "cmd_l1", "cmd_l2", "cmd_l3", "ind_l0", "ind_l1", "ind_l2", "ind_l3")
        x_all: dict[str, np.ndarray] = {}
        for c in range(4):
            m = (labs == c).astype(float)
            x_all[f"cmd_l{c}"] = cmds * m
            x_all[f"ind_l{c}"] = m
        n = len(trues)
        idx_tr = np.arange(0, n, 2)
        idx_va = np.arange(1, n, 2)
        tr_d = {"x": {k: v[idx_tr] for k, v in x_all.items()}, "y": trues[idx_tr]}
        va_d = {"x": {k: v[idx_va] for k, v in x_all.items()}, "y": trues[idx_va]}
        cal_tid = f"cal_alpha_{episode_id}"
        shared_id = "alpha_shared_loadcell_calibration"
        # Remove prior transient calibration task models + child mechanisms on same root so `AncestorReweight` fits current epoch cleanly
        for old_t in [t for t in list(graph.active_task_models.keys()) if t.startswith("cal_alpha_")]:
            graph.active_task_models.pop(old_t, None)
            graph.task_datasets.pop(old_t, None)
            graph.mechanisms[shared_id].dependent_task_ids.discard(old_t)
        for old_m in [m for m in list(graph.mechanisms.keys()) if "cal_alpha_" in m and m != shared_id]:
            graph.mechanisms.pop(old_m, None)
        planner.register_task(
            task_id=cal_tid,
            shared_upstream_id=shared_id,
            feature_names=cols,
            train_data=tr_d,
            val_data=va_d,
            initial_local_shrinkage=1e6,
        )
        rev = planner.evaluate_and_revise_task(cal_tid, budget_remaining=1.0)
        sel_act = str(rev["selected_candidate"]["action_type"])
        sw = graph.mechanisms[shared_id].shared_weights
        ver = int(graph.mechanisms[shared_id].version)
        payload = {
            "stress_gains": [float(sw[f"cmd_l{c}"]) for c in range(4)],
            "stress_biases": [float(sw[f"ind_l{c}"]) for c in range(4)],
        }
    else:
        st = public_item["study"]
        cal = st["sensor_calibration_sweeps"]
        banks = np.asarray(cal["bank_id"], dtype=int).ravel()
        refs = np.asarray(cal["ref_signal"], dtype=float).ravel()
        raws = np.asarray(cal["raw_readout"], dtype=float).ravel()
        cols = ("ref_b0", "ref_b1", "ref_b2", "ref_b3", "ind_b0", "ind_b1", "ind_b2", "ind_b3")
        x_all = {}
        for b in range(4):
            m = (banks == b).astype(float)
            x_all[f"ref_b{b}"] = refs * m
            x_all[f"ind_b{b}"] = m
        n = len(raws)
        idx_tr = np.arange(0, n, 2)
        idx_va = np.arange(1, n, 2)
        tr_d = {"x": {k: v[idx_tr] for k, v in x_all.items()}, "y": raws[idx_tr]}
        va_d = {"x": {k: v[idx_va] for k, v in x_all.items()}, "y": raws[idx_va]}
        cal_tid = f"cal_beta_{episode_id}"
        shared_id = "beta_shared_sensor_bank_calibration"
        for old_t in [t for t in list(graph.active_task_models.keys()) if t.startswith("cal_beta_")]:
            graph.active_task_models.pop(old_t, None)
            graph.task_datasets.pop(old_t, None)
            graph.mechanisms[shared_id].dependent_task_ids.discard(old_t)
        for old_m in [m for m in list(graph.mechanisms.keys()) if "cal_beta_" in m and m != shared_id]:
            graph.mechanisms.pop(old_m, None)
        planner.register_task(
            task_id=cal_tid,
            shared_upstream_id=shared_id,
            feature_names=cols,
            train_data=tr_d,
            val_data=va_d,
            initial_local_shrinkage=1e6,
        )
        rev = planner.evaluate_and_revise_task(cal_tid, budget_remaining=1.0)
        sel_act = str(rev["selected_candidate"]["action_type"])
        sw = graph.mechanisms[shared_id].shared_weights
        ver = int(graph.mechanisms[shared_id].version)
        payload = {
            "bank_gains": [float(sw[f"ref_b{b}"]) for b in range(4)],
            "bank_offsets": [float(sw[f"ind_b{b}"]) for b in range(4)],
        }

    updated_bids: list[str] = []
    if sel_act == "AncestorReweight":
        updated_bids = registry.update_shared_parameter_epoch(
            parameter_mechanism_id=shared_id,
            new_version=ver,
            parameter_payload=payload,
            affected_chain_id=chain_id,
            evidence_note=f"m8_ancestor_reweight_on_{episode_id}",
        )
    else:
        # Ensure current epoch payload stays synced
        registry.parameter_epochs[shared_id] = {
            "parameter_mechanism_id": shared_id,
            "version": ver,
            "payload": payload,
            "evidence_note": f"retained_on_{episode_id}",
        }
        registry.sync_to_graph()

    sentinel_after = np.asarray(evaluate_committed_task_model(sentinel_tm, graph, sentinel_va)["predictions"], dtype=float)
    sentinel_drift = float(np.max(np.abs(sentinel_after - sentinel_before)))
    return {
        "chain_id": chain_id,
        "episode_id": episode_id,
        "shared_mechanism_id": shared_id,
        "planner_selected_action": sel_act,
        "shared_mechanism_version": ver,
        "parameter_payload": payload,
        "updated_binding_ids": updated_bids,
        "val_log_predictive_gain": float(rev["selected_candidate"]["val_log_predictive_gain"]),
        "sentinel_max_abs_prediction_drift": sentinel_drift,
    }


def extract_guard_observables_from_m94_public_item(chain_id: str, public_item: dict[str, Any], param_payload: dict[str, Any]) -> dict[str, Any]:
    """Compute calibrated observable context for `ObservablePreconditionGuard` using ONLY public inputs + current `param_payload`."""
    if chain_id == "alpha":
        ds = public_item["dataset"]
        gains = np.asarray(param_payload.get("stress_gains", [1.0, 1.0, 1.0, 1.0]), dtype=float)
        biases = np.asarray(param_payload.get("stress_biases", [0.0, 0.0, 0.0, 0.0]), dtype=float)
        te_l = np.asarray(ds["test"]["lab_id"], dtype=int)
        tr_l = np.asarray(ds["train"]["lab_id"], dtype=int)
        cal_past_te = np.asarray(ds["test"]["past_stress_cmd"], dtype=float) * gains[te_l, None] + biases[te_l, None]
        cal_hold_te = np.asarray(ds["test"]["stress_hold_cmd"], dtype=float) * gains[te_l] + biases[te_l]
        cal_past_tr = np.asarray(ds["train"]["past_stress_cmd"], dtype=float) * gains[tr_l, None] + biases[tr_l, None]
        cal_hold_tr = np.asarray(ds["train"]["stress_hold_cmd"], dtype=float) * gains[tr_l] + biases[tr_l]
        obs: dict[str, Any] = {
            "past_stress": np.vstack([cal_past_tr, cal_past_te]),
            "stress_hold_0": np.concatenate([cal_hold_tr, cal_hold_te]),
            "train": {"past_stress": cal_past_tr, "stress_hold_0": cal_hold_tr},
            "test": {"past_stress": cal_past_te, "stress_hold_0": cal_hold_te},
            "eval_times": np.asarray(ds["eval_times"], dtype=float),
        }
        if "yield_stress" in ds:
            obs["yield_stress"] = float(ds["yield_stress"])
        elif ds.get("pure_linear_no_yield_declared") is True:
            obs["pure_linear_no_yield_declared"] = True
        if "residual_plastic_state" in ds.get("test", {}):
            obs["residual_plastic_state"] = np.asarray(ds["test"]["residual_plastic_state"], dtype=float)
        return obs
    st = public_item["study"]
    v_mat = st.get("V_channel_b")
    if v_mat is None:
        v_mat = st.get("V_background")
    if v_mat is None:
        return {"V_pocket_channel_b": np.zeros((int(st["nx"]), int(st["ny"])), dtype=float)}
    return {"V_pocket_channel_b": np.asarray(v_mat, dtype=float)}


def build_parameterized_source_closure_code(chain_id: str, entrypoint_name: str, param_payload: dict[str, Any]) -> str:
    """Emit a clean, self-contained parameterized Python closure bound to `(C0_base, param_version)`.

    NOTE: This closure embeds ONLY the `Stage S` constitutive/operator law + the versioned shared upstream calibration
    parameters (`param_payload`) from `PersistentMechanismGraph`. It NEVER embeds held-out test answers or `Stage C`
    nonlinear extensions, and it naturally fails schema/probe checks on `ep01` (`T` shifted readout) and `ep03` (`C`
    premise counterexample) while succeeding on `ep02` (`G`) and `ep04` (`R`)!
    """
    if chain_id == "alpha":
        gains_lit = [round(float(x), 8) for x in param_payload["stress_gains"]]
        biases_lit = [round(float(x), 8) for x in param_payload["stress_biases"]]
        return f'''"""Bound `M9.4` Parameterized `C0` Hereditary Mechanism Closure (`alpha_shared_loadcell_calibration`)."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

BOUND_STRESS_GAINS = np.array({gains_lit!r}, dtype=float)
BOUND_STRESS_BIASES = np.array({biases_lit!r}, dtype=float)


def _align_coupons(tr: dict, te: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labs = np.concatenate([tr["lab_id"], te["lab_id"]])
    strata = np.concatenate([tr["stratum"], te["stratum"]])
    X_raw = np.vstack([tr["coupon_features_raw"], te["coupon_features_raw"]])
    R_raw = np.vstack([tr["ref_coupons"], te["ref_coupons"]])
    offsets = np.zeros(4, dtype=float)
    gains = np.ones(4, dtype=float)
    for c in range(4):
        Rc = R_raw[labs == c]
        col_m = np.nanmean(Rc, axis=0)
        col_s = np.nanstd(Rc, axis=0, ddof=0)
        valid = np.isfinite(col_m) & np.isfinite(col_s) & (col_s > 1e-8)
        offsets[c] = float(np.median(col_m[valid]))
        gains[c] = float(np.median(col_s[valid]))
    X_std = (X_raw - offsets[labs, None]) / gains[labs, None]
    w_all = np.zeros(len(labs), dtype=float)
    for c in range(4):
        for s in range(3):
            m = (labs == c) & (strata == s)
            if np.any(m):
                w_all[m] = 1.0 / float(np.sum(m))
    w_all /= float(np.mean(w_all))
    for j in range(X_std.shape[1]):
        fin = np.isfinite(X_std[:, j])
        fill = float(np.sum(w_all[fin] * X_std[fin, j]) / max(float(np.sum(w_all[fin])), 1e-12))
        X_std[~fin, j] = fill
    ntr = len(tr["lab_id"])
    w_tr = w_all[:ntr] / float(np.mean(w_all[:ntr]))
    return X_std[:ntr], X_std[ntr:], w_tr


def _prop_mem(past_s: np.ndarray, past_dt: np.ndarray, taus: np.ndarray, init_m: np.ndarray | None = None) -> np.ndarray:
    z = np.zeros((past_s.shape[0], 2), dtype=float) if init_m is None else np.asarray(init_m, dtype=float).copy()
    for si, dt in enumerate(past_dt):
        dec = np.exp(-float(dt) / taus)
        z = z * dec[None, :] + (1.0 - dec[None, :]) * past_s[:, si:si + 1]
    return z


def {entrypoint_name}(dataset: dict) -> dict:
    tr, te = dataset["train"], dataset["test"]
    past_dt = np.asarray(dataset["past_dt"], dtype=float)
    eval_t = np.asarray(dataset["eval_times"], dtype=float)
    X_tr, X_te, _ = _align_coupons(tr, te)

    tr_lab = np.asarray(tr["lab_id"], dtype=int)
    te_lab = np.asarray(te["lab_id"], dtype=int)
    tr_ps = np.asarray(tr["past_stress_cmd"], dtype=float) * BOUND_STRESS_GAINS[tr_lab, None] + BOUND_STRESS_BIASES[tr_lab, None]
    tr_h0 = np.asarray(tr["stress_hold_cmd"], dtype=float) * BOUND_STRESS_GAINS[tr_lab] + BOUND_STRESS_BIASES[tr_lab]
    te_ps = np.asarray(te["past_stress_cmd"], dtype=float) * BOUND_STRESS_GAINS[te_lab, None] + BOUND_STRESS_BIASES[te_lab, None]
    te_h0 = np.asarray(te["stress_hold_cmd"], dtype=float) * BOUND_STRESS_GAINS[te_lab] + BOUND_STRESS_BIASES[te_lab]

    tr_im = tr.get("initial_memory_states")
    te_im = te.get("initial_memory_states")
    y_curves = np.asarray(tr["strain_curves_obs"], dtype=float) - np.asarray(tr["strain_0"], dtype=float)[:, None]
    y_rate = np.asarray(tr["instant_rate_obs"], dtype=float)
    n_tr, K = y_curves.shape

    def _fit_given_taus(t1: float, t2: float, irls_iters: int = 8) -> tuple[float, np.ndarray]:
        taus = np.array([t1, t2], dtype=float)
        z_tr = _prop_mem(tr_ps, past_dt, taus, tr_im)
        gap = tr_h0[:, None] - z_tr
        e1 = 1.0 - np.exp(-eval_t[None, :] / t1)
        e2 = 1.0 - np.exp(-eval_t[None, :] / t2)
        D_c = np.zeros((n_tr * K, 7), dtype=float)
        D_c[:, 0] = (gap[:, 0:1] * e1).ravel()
        D_c[:, 1] = (gap[:, 1:2] * e2).ravel()
        D_c[:, 2] = np.broadcast_to(eval_t[None, :], (n_tr, K)).ravel()
        for j in range(4):
            D_c[:, 3 + j] = (X_tr[:, j:j + 1] * eval_t[None, :]).ravel()
        D_r = np.column_stack([gap[:, 0] / t1, gap[:, 1] / t2, np.ones(n_tr), X_tr])
        D_all = np.vstack([D_c, 1.8 * D_r])
        y_all = np.concatenate([y_curves.ravel(), 1.8 * y_rate])
        w_r = np.ones_like(y_all)
        th = np.zeros(7, dtype=float)
        scale2 = (0.045 ** 2)
        for _ in range(irls_iters):
            sw = np.sqrt(w_r)
            th, *_ = np.linalg.lstsq(D_all * sw[:, None], y_all * sw, rcond=None)
            res = y_all - D_all @ th
            loss_c = float(np.sum(np.log1p((res[:n_tr * K] ** 2) / scale2)) + 0.5 * np.sum((res[n_tr * K:] ** 2) / (0.025 ** 2)))
            w_r[:n_tr * K] = 1.0 / (1.0 + (res[:n_tr * K] ** 2) / scale2)
            w_r[n_tr * K:] = 1.0 / (1.0 + 0.2 * (res[n_tr * K:] ** 2) / (0.025 ** 2))
        return loss_c, th

    best = (1e30, 0.6, 4.0, np.zeros(7))
    for t1 in np.geomspace(0.28, 1.40, 10):
        for t2 in np.geomspace(max(t1 * 1.8, 1.8), 6.8, 10):
            lv, th = _fit_given_taus(float(t1), float(t2), irls_iters=5)
            if lv < best[0]:
                best = (lv, float(t1), float(t2), th)

    res_opt = minimize(
        lambda u: _fit_given_taus(float(np.exp(u[0])), float(np.exp(u[0]) + np.exp(u[1])), irls_iters=8)[0],
        x0=np.array([np.log(best[1]), np.log(best[2] - best[1])]),
        method="Nelder-Mead",
        options={{"xatol": 1e-4, "fatol": 1e-4, "maxiter": 90}},
    )
    t1_hat = float(np.exp(res_opt.x[0]))
    t2_hat = float(t1_hat + np.exp(res_opt.x[1]))
    _, th_hat = _fit_given_taus(t1_hat, t2_hat, irls_iters=12)
    taus_hat = np.array([t1_hat, t2_hat], dtype=float)
    z_te = _prop_mem(te_ps, past_dt, taus_hat, te_im)
    gap_te = te_h0[:, None] - z_te
    drift_te = th_hat[2] + X_te @ th_hat[3:]
    pred_s = (
        np.asarray(te["strain_0"], dtype=float)[:, None]
        + th_hat[0] * gap_te[:, 0:1] * (1.0 - np.exp(-eval_t[None, :] / t1_hat))
        + th_hat[1] * gap_te[:, 1:2] * (1.0 - np.exp(-eval_t[None, :] / t2_hat))
        + drift_te[:, None] * eval_t[None, :]
    )
    rate_te = th_hat[0] * gap_te[:, 0] / t1_hat + th_hat[1] * gap_te[:, 1] / t2_hat + drift_te
    return {{
        "estimated_taus": taus_hat,
        "hereditary_weights": th_hat,
        "predicted_strain": pred_s,
        "initial_rate_pred": rate_te,
    }}
'''

    gains_b = [round(float(x), 8) for x in param_payload["bank_gains"]]
    offs_b = [round(float(x), 8) for x in param_payload["bank_offsets"]]
    return f'''"""Bound `M9.4` Parameterized `C0` Separable Kronecker + Box-Simplex KKT Closure (`beta_shared_sensor_bank_calibration`)."""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import minimize

BOUND_BANK_GAINS = np.array({gains_b!r}, dtype=float)
BOUND_BANK_OFFSETS = np.array({offs_b!r}, dtype=float)


def _build_1d_op(L: float, n: int, kappa: np.ndarray, vel: np.ndarray, rg: float, diag_add: np.ndarray | None = None) -> np.ndarray:
    h = float(L) / int(n)
    kap = np.asarray(kappa, dtype=float).ravel()
    v = np.asarray(vel, dtype=float).ravel()
    A = np.zeros((n, n), dtype=float)
    kh = (2.0 * kap[:-1] * kap[1:] / (kap[:-1] + kap[1:])) / (h * h)
    idx = np.arange(n - 1)
    A[idx, idx] += kh
    A[idx + 1, idx + 1] += kh
    A[idx, idx + 1] -= kh
    A[idx + 1, idx] -= kh
    A[0, 0] += (rg * kap[0]) / (h * (rg * 0.5 * h + kap[0]))
    A[n - 1, n - 1] += (rg * kap[n - 1]) / (h * (rg * 0.5 * h + kap[n - 1]))
    for i in range(n):
        vi = float(v[i])
        if vi >= 0.0:
            A[i, i] += vi / h
            if i > 0:
                A[i, i - 1] -= vi / h
        else:
            A[i, i] += (-vi) / h
            if i < n - 1:
                A[i, i + 1] -= (-vi) / h
    if diag_add is not None:
        A += np.diag(np.asarray(diag_add, dtype=float).ravel())
    return A


def _solve_box_simplex_kkt(S_tilde: np.ndarray, r_y: np.ndarray, lam_w: float, u_box: float, b_tot: float) -> tuple[np.ndarray, dict]:
    M = S_tilde.shape[1]
    H = S_tilde.T @ S_tilde + float(lam_w) * np.eye(M)
    f = S_tilde.T @ r_y
    w0 = np.full(M, float(b_tot) / M, dtype=float)
    res = minimize(
        lambda w: 0.5 * float(w @ (H @ w)) - float(f @ w),
        w0,
        jac=lambda w: H @ w - f,
        bounds=[(0.0, float(u_box))] * M,
        constraints=[{{"type": "eq", "fun": lambda w: float(np.sum(w)) - float(b_tot), "jac": lambda w: np.ones(M)}}],
        method="SLSQP",
        options={{"ftol": 1e-14, "maxiter": 400}},
    )
    w = np.clip(res.x, 0.0, float(u_box))
    lo, hi = float(np.min(w) - u_box), float(np.max(w))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(np.sum(np.clip(w - mid, 0.0, float(u_box)))) > float(b_tot):
            lo = mid
        else:
            hi = mid
    w = np.clip(w - 0.5 * (lo + hi), 0.0, float(u_box))
    for _ in range(8):
        free = (w > 1e-11) & (w < float(u_box) - 1e-11)
        f_idx = np.flatnonzero(free)
        if f_idx.size >= 2:
            rem = float(b_tot) - float(np.sum(w[~free]))
            Hf = H[np.ix_(f_idx, f_idx)]
            rhs_f = f[f_idx] - H[np.ix_(f_idx, np.flatnonzero(~free))] @ w[~free]
            aug = np.block([[Hf, -np.ones((f_idx.size, 1))], [np.ones((1, f_idx.size)), np.zeros((1, 1))]])
            sol = np.linalg.solve(aug, np.concatenate([rhs_f, [rem]]))
            wf = sol[:-1]
            if np.all((wf >= -1e-12) & (wf <= float(u_box) + 1e-12)):
                w[f_idx] = np.clip(wf, 0.0, float(u_box))
                break
    r_vec = S_tilde @ w - r_y
    j_val = 0.5 * float(r_vec @ r_vec) + 0.5 * float(lam_w) * float(w @ w)
    g = H @ w - f
    cand_nus = np.unique(g)
    best_nu = float(cand_nus[0])
    best_lb = -1e30
    for nu_c in cand_nus:
        lb_c = j_val - float(g @ w) + float(nu_c) * float(b_tot) + float(u_box) * float(np.sum(np.minimum(0.0, g - float(nu_c))))
        if lb_c > best_lb:
            best_lb = lb_c
            best_nu = float(nu_c)
    ll = np.maximum(0.0, g - best_nu)
    lu = np.maximum(0.0, best_nu - g)
    return w, {{
        "weights": w.tolist(),
        "nu_sum": float(best_nu),
        "lam_lower": ll.tolist(),
        "lam_upper": lu.tolist(),
        "primal_objective": float(j_val),
        "dual_lower_bound": float(min(best_lb, j_val)),
    }}


def {entrypoint_name}(study: dict) -> dict:
    if "budget_grid" in study or "V_channel_b" in study:
        raise ValueError("Bound single-channel single-budget C0 closure requires interface/branch adaptation.")
    nx, ny = int(study["nx"]), int(study["ny"])
    ax_diag = np.asarray(study.get("reaction_x", np.zeros(nx)), dtype=float).copy()
    by_diag = np.asarray(study.get("reaction_y", np.zeros(ny)), dtype=float).copy()
    if "V_background" in study and study["V_background"] is not None:
        V_bg = np.asarray(study["V_background"], dtype=float)
        # Exact additive separable 1D decomposition `V(x,y) = a_x(x) + b_y(y)` when guard verified separability!
        v_mean = float(np.mean(V_bg))
        ax_diag = ax_diag + np.mean(V_bg, axis=1)
        by_diag = by_diag + (np.mean(V_bg, axis=0) - v_mean)
    Ax = _build_1d_op(study["Lx"], nx, study["kappa_x"], study["vel_x"], float(study["robin_x"]), diag_add=ax_diag)
    By = _build_1d_op(study["Ly"], ny, study["kappa_y"], study["vel_y"], float(study["robin_y"]), diag_add=by_diag)
    mu_shifts = np.asarray(study["mu_shifts"], dtype=float)
    modes = np.asarray(study["source_modes"], dtype=float)
    si = np.asarray(study["sensor_ij"], dtype=int)[:, 0]
    sj = np.asarray(study["sensor_ij"], dtype=int)[:, 1]

    lx, Vx = np.linalg.eig(Ax)
    ly, Vy = np.linalg.eig(By)
    Phi_hat = np.einsum("ia,mab,bj->mij", np.linalg.inv(Vx), modes, np.linalg.inv(Vy).T)
    denom = lx[:, None, None] + ly[None, :, None] + mu_shifts[None, None, :]
    U_sep = np.real(np.einsum("ai,mijk,jb->mkab", Vx, Phi_hat[:, :, :, None] / denom[None, :, :, :], Vy.T))
    S_stack = np.vstack([U_sep[:, k, si, sj].T for k in range(len(mu_shifts))])

    raw_y = np.asarray(study["y_obs_raw"], dtype=float).ravel()
    row_banks = np.asarray(study["row_bank_id"], dtype=int).ravel()
    y_cal = (raw_y - BOUND_BANK_OFFSETS[row_banks]) / BOUND_BANK_GAINS[row_banks]

    Z = np.asarray(study["Z_nuisance"], dtype=float)
    Pz = np.linalg.solve(Z.T @ Z + float(study["nuisance_ridge"]) * np.eye(Z.shape[1]), Z.T)
    r_y = y_cal - Z @ (Pz @ y_cal)
    S_tilde = S_stack - Z @ (Pz @ S_stack)
    w, cert = _solve_box_simplex_kkt(S_tilde, r_y, float(study["source_ridge"]), float(study["u_box"]), float(study["total_budget"]))
    term_f = np.tensordot(w, U_sep[:, -1, :, :], axes=(0, 0))
    return {{
        "weights": w,
        "terminal_field": term_f,
        "kkt_certificate": cert,
    }}
'''


# ============================================================================
# 2. Continual World Generators (`Chain Alpha` & `Chain Beta`, Neutral IDs)
# ============================================================================

_M94_ALPHA_GEN_PY = r'''
def _gen_m94_alpha_cohort(
    seed,
    taus_pair,
    theta_vec,
    *,
    stress_gains=(1.0, 1.0, 1.0, 1.0),
    stress_biases=(0.0, 0.0, 0.0, 0.0),
    n_train=200,
    n_test=80,
    H=18,
    K=10,
    dt_mode="base_var",
    preload_scale=0.0,
    yield_stress=None,
    vp_theta=(0.0, 0.0),
    include_reset_metadata=False,
):
    rng = np.random.default_rng(seed)
    taus = np.asarray(taus_pair, dtype=float)
    theta = np.asarray(theta_vec, dtype=float)
    s_gains = np.asarray(stress_gains, dtype=float)
    s_biases = np.asarray(stress_biases, dtype=float)

    if dt_mode == "base_var":
        past_dt = np.round(rng.uniform(0.18, 0.85, size=H), 4)
        eval_times = np.round(np.linspace(0.25, 5.20, K), 4)
    else:
        past_dt = np.round(np.geomspace(0.09, 1.25, num=H) * rng.uniform(0.88, 1.12, size=H), 4)
        eval_times = np.round(np.geomspace(0.18, 6.40, num=K), 4)

    # Observable 4-lab load-cell calibration table (`40` reference checks across labs `0..3`)
    cal_labs = np.repeat(np.arange(4, dtype=int), 10)
    cal_cmd = rng.uniform(-2.2, 2.4, size=40)
    cal_true = s_gains[cal_labs] * cal_cmd + s_biases[cal_labs] + rng.normal(scale=1e-4, size=40)

    p_probs = np.array([[0.42, 0.33, 0.25], [0.22, 0.48, 0.30], [0.30, 0.24, 0.46], [0.26, 0.36, 0.38]])
    gains = np.array([0.82, 1.28, 0.94, 1.42])
    offsets = np.array([-0.65, 0.48, -0.25, 0.80])

    def _make_panel(n_rows):
        labs = rng.integers(0, 4, size=n_rows)
        strata = np.array([rng.choice(3, p=p_probs[c]) for c in labs], dtype=int)
        Z = rng.normal(loc=0.25 * (strata[:, None] - 1.0), scale=1.0, size=(n_rows, 4))
        R_lat = rng.normal(size=(n_rows, 32))
        X_raw = offsets[labs, None] + gains[labs, None] * Z
        R_raw = offsets[labs, None] + gains[labs, None] * R_lat
        X_raw[rng.random(X_raw.shape) < 0.025] = np.nan
        R_raw[rng.random(R_raw.shape) < 0.040] = np.nan
        return labs, strata, Z, X_raw, R_raw

    tr_lab, tr_st, tr_Z, tr_X, tr_R = _make_panel(n_train)
    te_lab, te_st, te_Z, te_X, te_R = _make_panel(n_test)

    def _sim(Z_lat, labs_arr, *, n_twins=0, add_glitches=False):
        N = Z_lat.shape[0]
        init_mem = rng.normal(scale=preload_scale, size=(N, 2)) if preload_scale > 0.0 else np.zeros((N, 2), dtype=float)
        hist_true = rng.normal(scale=1.55, size=(N, H))
        for s in range(1, H):
            hist_true[:, s] = 0.52 * hist_true[:, s - 1] + math.sqrt(1.0 - 0.52 ** 2) * hist_true[:, s]
        s0 = rng.normal(scale=0.35, size=N)
        hold_true = rng.uniform(-2.15, 2.35, size=N)
        if yield_stress is not None:
            if abs(float(vp_theta[0])) > 0.0 or abs(float(vp_theta[1])) > 0.0:
                hi_idx = np.arange(N // 2)
                sgn = np.where(rng.random(len(hi_idx)) < 0.55, 1.0, -1.0)
                hold_true[hi_idx] = sgn * rng.uniform(float(yield_stress) + 0.35, float(yield_stress) + 1.35, size=len(hi_idx))
                hist_true[hi_idx, -6:] = sgn[:, None] * rng.uniform(float(yield_stress) + 0.25, float(yield_stress) + 1.20, size=(len(hi_idx), 6))
                sub_idx = np.arange(N // 2, N)
                hold_true[sub_idx] = rng.uniform(-0.88 * float(yield_stress), 0.88 * float(yield_stress), size=len(sub_idx))
                hist_true[sub_idx] = np.clip(hist_true[sub_idx] * 0.52, -0.90 * float(yield_stress), 0.90 * float(yield_stress))
            else:
                hold_true = np.clip(hold_true, -0.86 * float(yield_stress), 0.86 * float(yield_stress))
                hist_true = np.clip(hist_true, -0.86 * float(yield_stress), 0.86 * float(yield_stress))

        if n_twins > 0:
            for k in range(n_twins):
                ia, ib = N - 2 * n_twins + 2 * k, N - 2 * n_twins + 2 * k + 1
                labs_arr[ib] = labs_arr[ia]
                Z_lat[ib] = Z_lat[ia]
                s0[ib] = s0[ia]
                hold_true[ib] = hold_true[ia]
                init_mem[ib] = -init_mem[ia]
                hist_true[ia] = np.linspace(1.9, -1.4, H)
                hist_true[ib] = np.linspace(-1.8, 1.6, H)
                if yield_stress is not None:
                    hist_true[ia] = np.clip(hist_true[ia], -0.88 * float(yield_stress), 0.88 * float(yield_stress))
                    hist_true[ib] = np.clip(hist_true[ib], -0.88 * float(yield_stress), 0.88 * float(yield_stress))
                    hold_true[ia] = hold_true[ib] = 0.75 * float(yield_stress)

        # Invert true physical stress to raw actuator commands per lab!
        hist_cmd = (hist_true - s_biases[labs_arr, None]) / s_gains[labs_arr, None]
        hold_cmd = (hold_true - s_biases[labs_arr]) / s_gains[labs_arr]

        z_cur = init_mem.copy()
        for s in range(H):
            dec = np.exp(-past_dt[s] / taus)
            z_cur = z_cur * dec[None, :] + (1.0 - dec[None, :]) * hist_true[:, s:s + 1]

        gap1 = hold_true - z_cur[:, 0]
        gap2 = hold_true - z_cur[:, 1]
        e1 = 1.0 - np.exp(-eval_times[None, :] / taus[0])
        e2 = 1.0 - np.exp(-eval_times[None, :] / taus[1])
        drift = theta[2] + Z_lat @ theta[3:7]

        vp_rate_hold = np.zeros(N, dtype=float)
        vp_dmg_surge = np.zeros(N, dtype=float)
        if yield_stress is not None:
            ys = float(yield_stress)
            ov_hold = (np.maximum(0.0, np.abs(hold_true) - ys) ** 2) * np.sign(hold_true)
            ov_hist = np.sum((np.maximum(0.0, np.abs(hist_true) - ys) ** 2) * past_dt[None, :], axis=1)
            vp_rate_hold = float(vp_theta[0]) * ov_hold
            vp_dmg_surge = float(vp_theta[1]) * ov_hist * hold_true

        clean = (
            s0[:, None]
            + theta[0] * gap1[:, None] * e1
            + theta[1] * gap2[:, None] * e2
            + (drift + vp_rate_hold)[:, None] * eval_times[None, :]
            + vp_dmg_surge[:, None] * e1
        )
        true_rate0 = theta[0] * gap1 / taus[0] + theta[1] * gap2 / taus[1] + drift + vp_rate_hold + vp_dmg_surge / taus[0]
        rate_obs = true_rate0 + rng.normal(scale=0.010, size=N)
        obs = clean + rng.normal(scale=0.009, size=clean.shape)
        if add_glitches:
            n_gl = max(1, int(0.08 * obs.size))
            g_flat = rng.choice(obs.size, size=n_gl, replace=False)
            obs.ravel()[g_flat] += rng.choice([-1.0, 1.0], size=n_gl) * rng.uniform(0.75, 1.15, size=n_gl)
        # Independent hazard threshold (ZERO future terminal strain correlation!)
        crit = s0 + rng.uniform(0.15, 0.58, size=N)
        ts_full = np.concatenate([[0.0], eval_times])
        ys_full = np.column_stack([s0, clean])
        cross = np.full(N, float(eval_times[-1]), dtype=float)
        for i in range(N):
            c = float(crit[i])
            if ys_full[i, 0] >= c:
                cross[i] = 0.0
                continue
            for ki in range(len(ts_full) - 1):
                if ys_full[i, ki + 1] >= c:
                    dy = max(ys_full[i, ki + 1] - ys_full[i, ki], 1e-12)
                    cross[i] = float(ts_full[ki] + (c - ys_full[i, ki]) / dy * (ts_full[ki + 1] - ts_full[ki]))
                    break
        secant = (clean[:, -1] - s0) / np.maximum(np.abs(hold_true), 0.5)
        return init_mem, hist_cmd, s0, hold_cmd, rate_obs, true_rate0, clean, obs, crit, cross, secant, hold_true, hist_true

    tr_im, tr_hc, tr_s0, tr_h0c, tr_ro, tr_rt, tr_cl, tr_ob, _, _, _, _, _ = _sim(tr_Z, tr_lab, n_twins=0, add_glitches=True)
    te_im, te_hc, te_s0, te_h0c, te_ro, te_rt, te_cl, _, te_cr, te_cx, te_sec, te_ht, te_hist_t = _sim(te_Z, te_lab, n_twins=8, add_glitches=False)
    for k in range(8):
        ia = len(te_Z) - 16 + 2 * k
        ib = ia + 1
        te_st[ib] = te_st[ia]
        te_X[ia] = offsets[te_lab[ia]] + gains[te_lab[ia]] * te_Z[ia]
        te_X[ib] = offsets[te_lab[ib]] + gains[te_lab[ib]] * te_Z[ib]

    ds = {
        "past_dt": past_dt,
        "eval_times": eval_times,
        "calibration_load_cells": {"lab_id": cal_labs, "cmd_stress": cal_cmd, "true_stress": cal_true},
        "train": {
            "lab_id": tr_lab, "stratum": tr_st, "coupon_features_raw": tr_X, "ref_coupons": tr_R,
            "past_stress_cmd": tr_hc, "strain_0": tr_s0, "stress_hold_cmd": tr_h0c, "instant_rate_obs": tr_ro,
            "strain_curves_obs": tr_ob,
        },
        "test": {
            "lab_id": te_lab, "stratum": te_st, "coupon_features_raw": te_X, "ref_coupons": te_R,
            "past_stress_cmd": te_hc, "strain_0": te_s0, "stress_hold_cmd": te_h0c, "instant_rate_obs": te_ro,
        },
    }
    if preload_scale > 0.0:
        ds["train"]["initial_memory_states"] = tr_im
        ds["test"]["initial_memory_states"] = te_im
        ds["test"]["critical_strain"] = te_cr
    if yield_stress is not None:
        ds["yield_stress"] = float(yield_stress)
    else:
        ds["pure_linear_no_yield_declared"] = True
    if include_reset_metadata:
        ds["specimen_reset"] = True
        ds["test"]["residual_plastic_state"] = np.zeros(len(te_lab), dtype=float)
    gt = {
        "taus": taus,
        "theta": theta,
        "vp_theta": np.asarray(vp_theta, dtype=float),
        "stress_gains": s_gains,
        "stress_biases": s_biases,
        "te_clean": te_cl,
        "te_true_rate0": te_rt,
        "te_cross": te_cx,
        "te_secant": te_sec,
        "te_hold_true": te_ht,
        "te_hist_true": te_hist_t,
    }
    return ds, gt
'''


_M94_BETA_GEN_PY = _SHARED_BOX_SIMPLEX_KKT_CHECKER_PY + r'''
def _build_1d_fvm_advdiff_robin(L, n, kappa, vel, robin_gamma, diag_add=None):
    h = float(L) / int(n)
    kap = np.asarray(kappa, dtype=float).ravel()
    v = np.asarray(vel, dtype=float).ravel()
    rg = float(robin_gamma)
    A = np.zeros((n, n), dtype=float)
    kh = (2.0 * kap[:-1] * kap[1:] / (kap[:-1] + kap[1:])) / (h * h)
    idx = np.arange(n - 1)
    A[idx, idx] += kh
    A[idx + 1, idx + 1] += kh
    A[idx, idx + 1] -= kh
    A[idx + 1, idx] -= kh
    A[0, 0] += (rg * kap[0]) / (h * (rg * 0.5 * h + kap[0]))
    A[n - 1, n - 1] += (rg * kap[n - 1]) / (h * (rg * 0.5 * h + kap[n - 1]))
    for i in range(n):
        vi = float(v[i])
        if vi >= 0.0:
            A[i, i] += vi / h
            if i > 0:
                A[i, i - 1] -= vi / h
        else:
            A[i, i] += (-vi) / h
            if i < n - 1:
                A[i, i + 1] -= (-vi) / h
    if diag_add is not None:
        A += np.diag(np.asarray(diag_add, dtype=float).ravel())
    return A

def _solve_box_simplex_exact(S_tilde, r_y, lam_w, u_box, b_tot):
    from scipy.optimize import minimize
    M = S_tilde.shape[1]
    H = S_tilde.T @ S_tilde + float(lam_w) * np.eye(M)
    f = S_tilde.T @ r_y
    w0 = np.full(M, float(b_tot) / M, dtype=float)
    res = minimize(
        lambda w: 0.5 * float(w @ (H @ w)) - float(f @ w),
        w0,
        jac=lambda w: H @ w - f,
        bounds=[(0.0, float(u_box))] * M,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w)) - float(b_tot), "jac": lambda w: np.ones(M)}],
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 400},
    )
    w = np.clip(res.x, 0.0, float(u_box))
    lo, hi = float(np.min(w) - u_box), float(np.max(w))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(np.sum(np.clip(w - mid, 0.0, float(u_box)))) > float(b_tot):
            lo = mid
        else:
            hi = mid
    w = np.clip(w - 0.5 * (lo + hi), 0.0, float(u_box))
    for _ in range(8):
        free = (w > 1e-11) & (w < float(u_box) - 1e-11)
        f_idx = np.flatnonzero(free)
        if f_idx.size >= 2:
            rem = float(b_tot) - float(np.sum(w[~free]))
            Hf = H[np.ix_(f_idx, f_idx)]
            rhs_f = f[f_idx] - H[np.ix_(f_idx, np.flatnonzero(~free))] @ w[~free]
            aug = np.block([[Hf, -np.ones((f_idx.size, 1))], [np.ones((1, f_idx.size)), np.zeros((1, 1))]])
            sol = np.linalg.solve(aug, np.concatenate([rhs_f, [rem]]))
            wf = sol[:-1]
            if np.all((wf >= -1e-12) & (wf <= float(u_box) + 1e-12)):
                w[f_idx] = np.clip(wf, 0.0, float(u_box))
                break
    r_vec = S_tilde @ w - r_y
    j_val = 0.5 * float(r_vec @ r_vec) + 0.5 * float(lam_w) * float(w @ w)
    g = H @ w - f
    cand_nus = np.unique(g)
    best_nu = float(cand_nus[0])
    best_lb = -1e30
    for nu_c in cand_nus:
        lb_c = j_val - float(g @ w) + float(nu_c) * float(b_tot) + float(u_box) * float(np.sum(np.minimum(0.0, g - float(nu_c))))
        if lb_c > best_lb:
            best_lb = lb_c
            best_nu = float(nu_c)
    ll = np.maximum(0.0, g - best_nu)
    lu = np.maximum(0.0, best_nu - g)
    return w, {
        "weights": w.tolist(),
        "nu_sum": float(best_nu),
        "lam_lower": ll.tolist(),
        "lam_upper": lu.tolist(),
        "primal_objective": float(j_val),
        "dual_lower_bound": float(min(best_lb, j_val)),
    }

def _gen_m94_beta_study(
    seed,
    *,
    nx=44,
    ny=40,
    K_mu=10,
    M=14,
    vel_sign=1.0,
    robin_pair=(1.35, 1.15),
    bank_gains=(1.0, 1.0, 1.0, 1.0),
    bank_offsets=(0.0, 0.0, 0.0, 0.0),
    multi_budget=False,
    dual_reactive_channels=False,
    additive_background=False,
):
    rng = np.random.default_rng(seed)
    b_gains = np.asarray(bank_gains, dtype=float)
    b_offs = np.asarray(bank_offsets, dtype=float)

    # Observable 4-bank sensor calibration sweep (`48` electronic reference calibration measurements)
    cal_banks = np.repeat(np.arange(4, dtype=int), 12)
    cal_ref = rng.uniform(-1.2, 1.6, size=48)
    cal_raw = b_gains[cal_banks] * cal_ref + b_offs[cal_banks] + rng.normal(scale=1e-8, size=48)

    Lx, Ly = 2.2, 1.9
    xs = (np.arange(nx) + 0.5) * (Lx / nx)
    ys = (np.arange(ny) + 0.5) * (Ly / ny)
    kappa_x = 0.42 + 0.16 * np.cos(math.pi * xs / Lx) + 0.05 * rng.uniform(-1.0, 1.0, size=nx)
    kappa_y = 0.38 + 0.15 * np.sin(math.pi * ys / Ly) + 0.05 * rng.uniform(-1.0, 1.0, size=ny)
    vel_x = float(vel_sign) * (0.48 + 0.12 * np.sin(2.0 * math.pi * xs / Lx))
    vel_y = float(vel_sign) * (-0.42 + 0.10 * np.cos(2.0 * math.pi * ys / Ly))
    rx_base = np.round(0.12 + 0.05 * np.cos(math.pi * xs / Lx), 6)
    ry_base = np.round(0.10 + 0.04 * np.sin(math.pi * ys / Ly), 6)

    V_bg = None
    ax_add = rx_base.copy()
    by_add = ry_base.copy()
    if additive_background or dual_reactive_channels:
        # Strictly ADDITIVE separable 2D background `V_bg(x,y) = vx(x) + vy(y) > 0` (`nonseparable_ratio == 0.0`!)
        vx_part = 0.35 + 0.18 * np.cos(2.0 * math.pi * xs / Lx)
        vy_part = 0.28 + 0.14 * np.sin(2.0 * math.pi * ys / Ly)
        V_bg = vx_part[:, None] + vy_part[None, :]
        ax_add = rx_base + np.mean(V_bg, axis=1)
        by_add = ry_base + (np.mean(V_bg, axis=0) - float(np.mean(V_bg)))

    Ax_eff = _build_1d_fvm_advdiff_robin(Lx, nx, kappa_x, vel_x, robin_pair[0], diag_add=ax_add)
    By_eff = _build_1d_fvm_advdiff_robin(Ly, ny, kappa_y, vel_y, robin_pair[1], diag_add=by_add)
    Ax_base = _build_1d_fvm_advdiff_robin(Lx, nx, kappa_x, vel_x, robin_pair[0], diag_add=rx_base)
    By_base = _build_1d_fvm_advdiff_robin(Ly, ny, kappa_y, vel_y, robin_pair[1], diag_add=ry_base)
    mu_shifts = np.round(np.geomspace(0.08, 3.40, num=K_mu), 4)

    X2, Y2 = np.meshgrid(xs, ys, indexing="ij")
    centers_x = rng.uniform(0.18 * Lx, 0.82 * Lx, size=M)
    centers_y = rng.uniform(0.18 * Ly, 0.82 * Ly, size=M)
    source_modes = np.zeros((M, nx, ny), dtype=float)
    for m in range(M):
        g_m = np.exp(-((X2 - centers_x[m]) ** 2 + (Y2 - centers_y[m]) ** 2) / (2.0 * (0.21 ** 2)))
        source_modes[m] = g_m / max(float(np.linalg.norm(g_m)), 1e-12)

    N_sens = 36
    si = rng.integers(2, nx - 2, size=N_sens)
    sj = rng.integers(2, ny - 2, size=N_sens)
    sensor_ij = np.column_stack([si, sj])
    sens_bank = np.arange(N_sens, dtype=int) % 4

    lx, Vx = np.linalg.eig(Ax_eff)
    ly, Vy = np.linalg.eig(By_eff)
    Phi_hat = np.einsum("ia,mab,bj->mij", np.linalg.inv(Vx), source_modes, np.linalg.inv(Vy).T)
    denom = lx[:, None, None] + ly[None, :, None] + mu_shifts[None, None, :]
    U_sep = np.real(np.einsum("ai,mijk,jb->mkab", Vx, Phi_hat[:, :, :, None] / denom[None, :, :, :], Vy.T))
    S_sep = np.vstack([U_sep[:, k, si, sj].T for k in range(K_mu)])

    V_chB = None
    U_chB = None
    if dual_reactive_channels:
        from scipy import sparse
        from scipy.sparse.linalg import splu
        V_chB = (
            4.8 * np.exp(-((X2 - 0.52 * Lx) ** 2 + (Y2 - 0.48 * Ly) ** 2) / 0.14) * np.sin(2.6 * math.pi * X2 / Lx) * np.sin(2.4 * math.pi * Y2 / Ly)
            + 3.4 * np.exp(-((X2 - 0.35 * Lx - 0.45 * Y2) ** 2) / 0.08)
        )
        V_chB = V_chB - float(np.min(V_chB)) + 0.15
        K_base = sparse.kron(sparse.csc_matrix(Ax_base), sparse.eye(ny, format="csc"), format="csc") + sparse.kron(sparse.eye(nx, format="csc"), sparse.csc_matrix(By_base), format="csc") + sparse.diags(V_chB.ravel(), format="csc")
        U_chB = np.zeros((M, K_mu, nx, ny), dtype=float)
        rhs_mat = source_modes.reshape(M, nx * ny).T
        for k in range(K_mu):
            lu_k = splu((K_base + float(mu_shifts[k]) * sparse.eye(nx * ny, format="csc")).tocsc())
            U_chB[:, k, :, :] = lu_k.solve(rhs_mat).T.reshape(M, nx, ny)
        S_chB = np.vstack([U_chB[:, k, si, sj].T for k in range(K_mu)])
        S_stack = np.vstack([S_sep, S_chB])
        row_bank_id = np.tile(sens_bank, 2 * K_mu)
    else:
        S_stack = S_sep
        row_bank_id = np.tile(sens_bank, K_mu)

    n_rows = S_stack.shape[0]
    P_z = 5
    Z_nuis = rng.normal(size=(n_rows, P_z))
    lam_z = 0.65
    lam_w = 0.04
    u_box = 1.0
    b_tot = 2.60

    w_true = np.zeros(M, dtype=float)
    act_m = [1, 4, 8, 11]
    raw_w = np.array([0.88, 0.72, 0.58, 0.42], dtype=float)
    w_true[act_m] = raw_w * (b_tot / float(np.sum(raw_w)))
    gamma_true = rng.normal(scale=0.65, size=P_z)
    y_true_cal = S_stack @ w_true + Z_nuis @ gamma_true + rng.normal(scale=0.012, size=n_rows)
    y_obs_raw = b_gains[row_bank_id] * y_true_cal + b_offs[row_bank_id]

    Pz_mat = np.linalg.solve(Z_nuis.T @ Z_nuis + lam_z * np.eye(P_z), Z_nuis.T)
    r_y = y_true_cal - Z_nuis @ (Pz_mat @ y_true_cal)
    S_tilde = S_stack - Z_nuis @ (Pz_mat @ S_stack)

    study = {
        "Lx": Lx, "Ly": Ly, "nx": nx, "ny": ny,
        "kappa_x": kappa_x, "vel_x": vel_x, "robin_x": float(robin_pair[0]), "reaction_x": rx_base,
        "kappa_y": kappa_y, "vel_y": vel_y, "robin_y": float(robin_pair[1]), "reaction_y": ry_base,
        "mu_shifts": mu_shifts,
        "source_modes": source_modes,
        "sensor_ij": sensor_ij,
        "row_bank_id": row_bank_id,
        "sensor_calibration_sweeps": {"bank_id": cal_banks, "ref_signal": cal_ref, "raw_readout": cal_raw},
        "Z_nuisance": Z_nuis,
        "y_obs_raw": y_obs_raw,
        "nuisance_ridge": lam_z,
        "source_ridge": lam_w,
        "u_box": u_box,
    }
    if additive_background:
        study["V_background"] = V_bg
    if multi_budget:
        budget_grid = np.array([1.40, 2.60, 3.80], dtype=float)
        study["budget_grid"] = budget_grid
        w_grid, certs_grid = [], []
        for bg in budget_grid:
            wb, cb = _solve_box_simplex_exact(S_tilde, r_y, lam_w, u_box, float(bg))
            w_grid.append(wb); certs_grid.append(cb)
        term_field = np.tensordot(w_grid[1], U_sep[:, -1, :, :], axes=(0, 0))
        gt = {
            "S_tilde": S_tilde, "r_y": r_y,
            "weights_by_budget": np.vstack(w_grid), "certs_by_budget": certs_grid,
            "terminal_field_mid_budget": term_field,
        }
    else:
        study["total_budget"] = b_tot
        if dual_reactive_channels:
            study["V_channel_a"] = V_bg
            study["V_channel_b"] = V_chB
        w_opt, cert_opt = _solve_box_simplex_exact(S_tilde, r_y, lam_w, u_box, b_tot)
        term_A = np.tensordot(w_opt, U_sep[:, -1, :, :], axes=(0, 0))
        term_B = np.tensordot(w_opt, U_chB[:, -1, :, :], axes=(0, 0)) if dual_reactive_channels else None
        gt = {
            "S_tilde": S_tilde, "r_y": r_y,
            "w_opt": w_opt, "cert_opt": cert_opt,
            "terminal_field_A": term_A, "terminal_field_B": term_B,
        }
    return study, gt
'''


# ============================================================================
# 3. 10 M9.4 Tasks (`2 Sources + 8 Continual Stream Tasks`, Neutral Task IDs!)
# ============================================================================

ALPHA_G_GAINS = "(1.28, 0.82, 1.18, 0.88)"
ALPHA_G_BIASES = "(0.14, -0.12, 0.09, -0.11)"
BETA_G_GAINS = "(1.24, 0.84, 1.15, 0.89)"
BETA_G_OFFS = "(0.08, -0.06, 0.05, -0.07)"


def _make_alpha_task(
    *,
    task_id: str,
    short_name: str,
    wave_label: str,
    artifact_name: str,
    entry_name: str,
    instruction_text: str,
    pub_call_expr: str,
    priv_cases_py: str,
    mode: str,  # 'standard' | 'transfer_T' | 'counterexample_C'
) -> HoldoutOODTask:
    pkl_name = f"public_{short_name}.pkl"
    if mode == "standard":
        schema_keys = '(("estimated_taus", (2,)), ("hereditary_weights", (7,)), ("predicted_strain", (n_te, k_t)), ("initial_rate_pred", (n_te,)))'
        priv_body = f'''
t0 = time.monotonic()
valid = True; max_tau_rel = 0.0; max_mae = 0.0; max_twin_mae = 0.0; max_rate_mae = 0.0; rows = []
for sd, ds, gt in cases:
    out = cand_mod.{entry_name}(ds)
    if not isinstance(out, dict) or not {{"estimated_taus", "hereditary_weights", "predicted_strain", "initial_rate_pred"}} <= set(out.keys()):
        valid = False; rows.append({{"seed": sd, "error": "missing_keys"}}); continue
    et = np.asarray(out["estimated_taus"], dtype=float)
    hw = np.asarray(out["hereditary_weights"], dtype=float)
    ps = np.asarray(out["predicted_strain"], dtype=float)
    ir = np.asarray(out["initial_rate_pred"], dtype=float)
    if et.shape != (2,) or hw.shape != (7,) or ps.shape != gt["te_clean"].shape or ir.shape != gt["te_true_rate0"].shape:
        valid = False; rows.append({{"seed": sd, "error": "shape_mismatch"}}); continue
    if not (np.all(np.isfinite(et)) and np.all(np.isfinite(hw)) and np.all(np.isfinite(ps)) and np.all(np.isfinite(ir)) and 0.05 < et[0] < et[1]):
        valid = False; rows.append({{"seed": sd, "error": "non_finite"}}); continue
    if float(np.linalg.norm(hw[:2])) <= 1e-4 or float(np.min(hw[:2])) <= 1e-4:
        valid = False; rows.append({{"seed": sd, "error": "degenerate_prony_weights"}}); continue
    z_m = np.zeros((len(ds["test"]["lab_id"]), 2), dtype=float)
    for s_i, dts in enumerate(ds["past_dt"]):
        dec = np.exp(-dts / et); z_m = z_m * dec[None, :] + (1.0 - dec[None, :]) * gt["te_hist_true"][:, s_i:s_i+1]
    gap_m = gt["te_hold_true"][:, None] - z_m
    pr_p = hw[0] * gap_m[:, 0:1] * (-np.expm1(-ds["eval_times"][None, :] / et[0])) + hw[1] * gap_m[:, 1:2] * (-np.expm1(-ds["eval_times"][None, :] / et[1]))
    rem_c = (ps - ds["test"]["strain_0"][:, None]) - pr_p
    rem_r = ir - (hw[0] * gap_m[:, 0] / et[0] + hw[1] * gap_m[:, 1] / et[1])
    if float(np.max(np.abs(rem_c - rem_r[:, None] * ds["eval_times"][None, :]))) > 0.08:
        valid = False; rows.append({{"seed": sd, "error": "weights_prediction_handoff_mismatch"}}); continue
    if float(np.max(np.abs(hw - gt["theta"]))) > 0.14:
        valid = False; rows.append({{"seed": sd, "error": "tail_coefficients_gt_handoff_mismatch"}}); continue
    tau_rel = float(np.max(np.abs(et - gt["taus"]) / gt["taus"]))
    s_mae = float(np.mean(np.abs(ps - gt["te_clean"])))
    tw_mae = float(np.mean(np.abs(ps[-16:] - gt["te_clean"][-16:])))
    r_mae = float(np.mean(np.abs(ir - gt["te_true_rate0"])))
    max_tau_rel = max(max_tau_rel, tau_rel); max_mae = max(max_mae, s_mae); max_twin_mae = max(max_twin_mae, tw_mae); max_rate_mae = max(max_rate_mae, r_mae)
    rows.append({{"seed": sd, "tau_rel": round(tau_rel, 6), "strain_mae": round(s_mae, 6), "twin_mae": round(tw_mae, 6), "rate_mae": round(r_mae, 6)}})
el = time.monotonic() - t0
sp = bool(valid and len(rows) == 2 and max_tau_rel <= 0.10 and max_mae <= 0.065 and max_twin_mae <= 0.065 and max_rate_mae <= 0.040 and el <= 20.0)
print(json.dumps({{"strict_pass": sp, "reward": 1.0 if sp else 0.0, "max_tau_rel_err": round(max_tau_rel, 6), "max_strain_mae": round(max_mae, 6), "max_twin_mae": round(max_twin_mae, 6), "max_rate_mae": round(max_rate_mae, 6), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}}, allow_nan=False))
'''
    elif mode == "transfer_T":
        schema_keys = '(("estimated_taus", (2,)), ("predicted_strain", (n_te, k_t)), ("secant_compliance", (n_te,)), ("hazard_crossing_times", (n_te,)))'
        priv_body = f'''
t0 = time.monotonic()
valid = True; max_tau = 0.0; max_s = 0.0; max_sec = 0.0; max_cx = 0.0; rows = []
for sd, ds, gt in cases:
    out = cand_mod.{entry_name}(ds)
    if not isinstance(out, dict) or not {{"estimated_taus", "predicted_strain", "secant_compliance", "hazard_crossing_times"}} <= set(out.keys()):
        valid = False; rows.append({{"seed": sd, "error": "missing_keys"}}); continue
    et = np.asarray(out["estimated_taus"], dtype=float)
    ps = np.asarray(out["predicted_strain"], dtype=float)
    sc = np.asarray(out["secant_compliance"], dtype=float)
    cx = np.asarray(out["hazard_crossing_times"], dtype=float)
    if et.shape != (2,) or ps.shape != gt["te_clean"].shape or sc.shape != gt["te_secant"].shape or cx.shape != gt["te_cross"].shape:
        valid = False; rows.append({{"seed": sd, "error": "shape_mismatch"}}); continue
    if not (np.all(np.isfinite(et)) and np.all(np.isfinite(ps)) and np.all(np.isfinite(sc)) and np.all(np.isfinite(cx)) and 0.05 < et[0] < et[1]):
        valid = False; rows.append({{"seed": sd, "error": "non_finite"}}); continue
    sec_exp = (ps[:, -1] - ds["test"]["strain_0"]) / np.maximum(np.abs(gt["te_hold_true"]), 0.5)
    if float(np.max(np.abs(sc - sec_exp))) > 1e-3:
        valid = False; rows.append({{"seed": sd, "error": "secant_handoff_mismatch"}}); continue
    tr = float(np.max(np.abs(et - gt["taus"]) / gt["taus"]))
    sm = float(np.mean(np.abs(ps - gt["te_clean"])))
    scm = float(np.mean(np.abs(sc - gt["te_secant"])))
    cxm = float(np.mean(np.abs(cx - gt["te_cross"])))
    max_tau = max(max_tau, tr); max_s = max(max_s, sm); max_sec = max(max_sec, scm); max_cx = max(max_cx, cxm)
    rows.append({{"seed": sd, "tau_rel": round(tr, 6), "strain_mae": round(sm, 6), "secant_mae": round(scm, 6), "cross_mae": round(cxm, 6)}})
el = time.monotonic() - t0
sp = bool(valid and len(rows) == 2 and max_tau <= 0.10 and max_s <= 0.075 and max_sec <= 0.095 and max_cx <= 0.28 and el <= 20.0)
print(json.dumps({{"strict_pass": sp, "reward": 1.0 if sp else 0.0, "max_tau_rel_err": round(max_tau, 6), "max_strain_mae": round(max_s, 6), "max_secant_mae": round(max_sec, 6), "max_crossing_mae": round(max_cx, 6), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}}, allow_nan=False))
'''
    else:
        schema_keys = '(("estimated_taus", (2,)), ("viscoplastic_weights", (2,)), ("predicted_strain", (n_te, k_t)))'
        priv_body = f'''
t0 = time.monotonic()
valid = True; max_tau = 0.0; max_vp = 0.0; max_all = 0.0; max_over = 0.0; max_sub = 0.0; rows = []
for sd, ds, gt in cases:
    out = cand_mod.{entry_name}(ds)
    if not isinstance(out, dict) or not {{"estimated_taus", "viscoplastic_weights", "predicted_strain"}} <= set(out.keys()):
        valid = False; rows.append({{"seed": sd, "error": "missing_keys"}}); continue
    et = np.asarray(out["estimated_taus"], dtype=float)
    vw = np.asarray(out["viscoplastic_weights"], dtype=float)
    ps = np.asarray(out["predicted_strain"], dtype=float)
    if et.shape != (2,) or vw.shape != (2,) or ps.shape != gt["te_clean"].shape:
        valid = False; rows.append({{"seed": sd, "error": "shape_mismatch"}}); continue
    if not (np.all(np.isfinite(et)) and np.all(np.isfinite(vw)) and np.all(np.isfinite(ps)) and 0.05 < et[0] < et[1]):
        valid = False; rows.append({{"seed": sd, "error": "non_finite"}}); continue
    ys_t = float(ds["yield_stress"])
    over_mask = (np.abs(gt["te_hold_true"]) > ys_t) | (np.max(np.abs(gt["te_hist_true"]), axis=1) > ys_t)
    sub_mask = ~over_mask
    tr = float(np.max(np.abs(et - gt["taus"]) / gt["taus"]))
    vr = float(np.linalg.norm(vw - gt["vp_theta"]) / np.linalg.norm(gt["vp_theta"]))
    am = float(np.mean(np.abs(ps - gt["te_clean"])))
    om = float(np.mean(np.abs(ps[over_mask] - gt["te_clean"][over_mask])))
    sm = float(np.mean(np.abs(ps[sub_mask] - gt["te_clean"][sub_mask])))
    max_tau = max(max_tau, tr); max_vp = max(max_vp, vr); max_all = max(max_all, am); max_over = max(max_over, om); max_sub = max(max_sub, sm)
    rows.append({{"seed": sd, "tau_rel": round(tr, 6), "vp_rel_l2": round(vr, 6), "all_mae": round(am, 6), "over_mae": round(om, 6), "sub_mae": round(sm, 6)}})
el = time.monotonic() - t0
sp = bool(valid and len(rows) == 2 and max_tau <= 0.10 and max_vp <= 0.14 and max_all <= 0.070 and max_over <= 0.080 and max_sub <= 0.065 and el <= 20.0)
print(json.dumps({{"strict_pass": sp, "reward": 1.0 if sp else 0.0, "max_tau_rel_err": round(max_tau, 6), "max_vp_rel_l2": round(max_vp, 6), "max_all_mae": round(max_all, 6), "max_overstress_mae": round(max_over, 6), "max_subyield_mae": round(max_sub, 6), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}}, allow_nan=False))
'''

    mod_stem = Path(artifact_name).stem
    return HoldoutOODTask(
        task_id=task_id,
        short_name=short_name,
        wave=wave_label,
        scientific_domain="Multi-Lab Polymer Coupon Hereditary Creep & Load-Cell Calibration Stream",
        wormhole_family="fam_hereditary_viscoelastic_memory",
        declared_artifacts=(f"/holdout_task/{artifact_name}",),
        declared_callable_entrypoints=(entry_name,),
        instruction_text=instruction_text,
        public_setup_py=(
            "import json, math, os, pickle, sys\nimport numpy as np\n"
            + _M94_ALPHA_GEN_PY
            + f"\npub_ds, pub_gt = {pub_call_expr}\n"
            f'with open("/holdout_task/{pkl_name}", "wb") as fh:\n'
            '    pickle.dump([{"dataset": pub_ds, "gt": pub_gt}], fh)\n'
            f'open("/holdout_task/validate_public_schema.py", "w").write("""import importlib.util, json, os, pickle, sys\n'
            "import numpy as np\n"
            "_d = os.path.dirname(os.path.abspath(__file__))\n"
            f'cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "{artifact_name}")\n'
            'spec = importlib.util.spec_from_file_location("_cand", cand_p)\n'
            "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
            f'ds = pickle.load(open(os.path.join(_d, "{pkl_name}"), "rb"))[0]["dataset"]\n'
            f"out = mod.{entry_name}(ds)\n"
            'n_te = len(ds["test"]["lab_id"]); k_t = len(ds["eval_times"])\n'
            "errs = []\n"
            'if not isinstance(out, dict): errs.append("not_dict")\n'
            "else:\n"
            f"    for k, shp in {schema_keys}:\n"
            '        if k not in out: errs.append(f"missing:{k}")\n'
            "        else:\n"
            "            a = np.asarray(out[k], dtype=float)\n"
            '            if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")\n'
            '    if not errs and "estimated_taus" in out:\n'
            '        ts = np.asarray(out["estimated_taus"], dtype=float)\n'
            '        if not (0.05 < ts[0] < ts[1] < 25.0): errs.append("taus_not_positive_ordered")\n'
            'print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))\n'
            "sys.exit(0 if len(errs) == 0 else 1)\n"
            '""")\n'
        ),
        private_verifier_py=(
            "import json, math, os, sys, time\nimport numpy as np\nsys.path.insert(0, '/holdout_task')\n"
            f"import {mod_stem} as cand_mod\n"
            + _M94_ALPHA_GEN_PY
            + f"\ncases = [\n{priv_cases_py}\n]\n"
            + priv_body
        ),
    )


def _make_beta_task(
    *,
    task_id: str,
    short_name: str,
    wave_label: str,
    artifact_name: str,
    entry_name: str,
    instruction_text: str,
    pub_call_expr: str,
    priv_cases_py: str,
    mode: str,  # 'standard' | 'multibudget_T' | 'dual_C'
) -> HoldoutOODTask:
    pkl_name = f"public_{short_name}.pkl"
    mod_stem = Path(artifact_name).stem
    if mode == "standard":
        val_check = (
            '    for k, shp in (("weights", (M,)), ("terminal_field", (nx, ny))):\n'
            '        if k not in out: errs.append(f"missing:{k}")\n'
            "        else:\n"
            "            a = np.asarray(out[k], dtype=float)\n"
            '            if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")\n'
            '    chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], st["total_budget"], out.get("kkt_certificate"), gap_tol=2e-4)\n'
            '    if not chk["valid"]: errs.append(f"kkt_invalid:{chk[\'reason\']}")\n'
            '    elif "weights" in out and float(np.max(np.abs(np.asarray(out["weights"], dtype=float) - np.asarray(out["kkt_certificate"]["weights"], dtype=float)))) > 1e-5: errs.append("weights_cert_mismatch")\n'
        )
        priv_body = f'''
t0 = time.monotonic()
valid = True; max_w_rel = 0.0; max_f_rel = 0.0; max_gap = 0.0; rows = []
for sd, st, gt in cases:
    out = cand_mod.{entry_name}(st)
    if not isinstance(out, dict) or not {{"weights", "terminal_field", "kkt_certificate"}} <= set(out.keys()):
        valid = False; rows.append({{"seed": sd, "error": "missing_keys"}}); continue
    w = np.asarray(out["weights"], dtype=float)
    tf = np.asarray(out["terminal_field"], dtype=float)
    if w.shape != gt["w_opt"].shape or tf.shape != gt["terminal_field_A"].shape or not (np.all(np.isfinite(w)) and np.all(np.isfinite(tf))):
        valid = False; rows.append({{"seed": sd, "error": "shape_or_nan"}}); continue
    chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], st["total_budget"], out["kkt_certificate"], gap_tol=2e-4)
    if not chk["valid"] or float(np.max(np.abs(w - np.asarray(out["kkt_certificate"]["weights"], dtype=float)))) > 1e-5:
        valid = False; rows.append({{"seed": sd, "error": f"kkt:{{chk.get('reason')}}"}}); continue
    wr = float(np.linalg.norm(w - gt["w_opt"]) / np.linalg.norm(gt["w_opt"]))
    fr = float(np.linalg.norm(tf - gt["terminal_field_A"]) / np.linalg.norm(gt["terminal_field_A"]))
    max_w_rel = max(max_w_rel, wr); max_f_rel = max(max_f_rel, fr); max_gap = max(max_gap, float(chk["gap"]))
    rows.append({{"seed": sd, "w_rel_l2": round(wr, 8), "field_rel_l2": round(fr, 8), "kkt_gap": round(float(chk["gap"]), 10)}})
el = time.monotonic() - t0
sp = bool(valid and len(rows) == 2 and max_w_rel <= 2e-4 and max_f_rel <= 2e-4 and max_gap <= 1e-5 and el <= 18.0)
print(json.dumps({{"strict_pass": sp, "reward": 1.0 if sp else 0.0, "max_weights_rel_l2": round(max_w_rel, 8), "max_field_rel_l2": round(max_f_rel, 8), "max_kkt_gap": round(max_gap, 10), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}}, allow_nan=False))
'''
    elif mode == "multibudget_T":
        val_check = (
            '    B = len(st["budget_grid"])\n'
            '    for k, shp in (("weights_by_budget", (B, M)), ("terminal_field_mid_budget", (nx, ny))):\n'
            '        if k not in out: errs.append(f"missing:{k}")\n'
            "        else:\n"
            "            a = np.asarray(out[k], dtype=float)\n"
            '            if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")\n'
            '    certs = out.get("kkt_certificates")\n'
            '    if not isinstance(certs, list) or len(certs) != B: errs.append("bad_certs_list")\n'
            "    elif not errs:\n"
            '        wb = np.asarray(out["weights_by_budget"], dtype=float)\n'
            '        for bi, bg in enumerate(st["budget_grid"]):\n'
            '            chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], float(bg), certs[bi], gap_tol=2e-4)\n'
            '            if not chk["valid"]: errs.append(f"kkt_b{bi}:{chk[\'reason\']}")\n'
            '            elif float(np.max(np.abs(wb[bi] - np.asarray(certs[bi]["weights"], dtype=float)))) > 1e-5: errs.append(f"wb_cert_mismatch_b{bi}")\n'
        )
        priv_body = f'''
t0 = time.monotonic()
valid = True; max_w_rel = 0.0; max_f_rel = 0.0; max_gap = 0.0; rows = []
for sd, st, gt in cases:
    out = cand_mod.{entry_name}(st)
    if not isinstance(out, dict) or not {{"weights_by_budget", "terminal_field_mid_budget", "kkt_certificates"}} <= set(out.keys()):
        valid = False; rows.append({{"seed": sd, "error": "missing_keys"}}); continue
    wb = np.asarray(out["weights_by_budget"], dtype=float)
    tf = np.asarray(out["terminal_field_mid_budget"], dtype=float)
    certs = out["kkt_certificates"]
    if wb.shape != gt["weights_by_budget"].shape or tf.shape != gt["terminal_field_mid_budget"].shape or not isinstance(certs, list) or len(certs) != 3:
        valid = False; rows.append({{"seed": sd, "error": "shape_mismatch"}}); continue
    if not (np.all(np.isfinite(wb)) and np.all(np.isfinite(tf))):
        valid = False; rows.append({{"seed": sd, "error": "non_finite"}}); continue
    ok_all = True
    for bi, bg in enumerate(st["budget_grid"]):
        chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], float(bg), certs[bi], gap_tol=2e-4)
        if not chk["valid"] or float(np.max(np.abs(wb[bi] - np.asarray(certs[bi]["weights"], dtype=float)))) > 1e-5:
            ok_all = False; rows.append({{"seed": sd, "error": f"kkt_b{{bi}}"}}); break
        max_gap = max(max_gap, float(chk["gap"]))
    if not ok_all:
        valid = False; continue
    wr = float(np.linalg.norm(wb - gt["weights_by_budget"]) / np.linalg.norm(gt["weights_by_budget"]))
    fr = float(np.linalg.norm(tf - gt["terminal_field_mid_budget"]) / np.linalg.norm(gt["terminal_field_mid_budget"]))
    max_w_rel = max(max_w_rel, wr); max_f_rel = max(max_f_rel, fr)
    rows.append({{"seed": sd, "w_rel_l2": round(wr, 8), "field_rel_l2": round(fr, 8), "max_kkt_gap": round(max_gap, 10)}})
el = time.monotonic() - t0
sp = bool(valid and len(rows) == 2 and max_w_rel <= 2e-4 and max_f_rel <= 2e-4 and max_gap <= 1e-5 and el <= 18.0)
print(json.dumps({{"strict_pass": sp, "reward": 1.0 if sp else 0.0, "max_weights_rel_l2": round(max_w_rel, 8), "max_field_rel_l2": round(max_f_rel, 8), "max_kkt_gap": round(max_gap, 10), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}}, allow_nan=False))
'''
    else:
        val_check = (
            '    for k, shp in (("weights", (M,)), ("terminal_field_channel_a", (nx, ny)), ("terminal_field_channel_b", (nx, ny))):\n'
            '        if k not in out: errs.append(f"missing:{k}")\n'
            "        else:\n"
            "            a = np.asarray(out[k], dtype=float)\n"
            '            if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")\n'
            '    chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], st["total_budget"], out.get("kkt_certificate"), gap_tol=2e-4)\n'
            '    if not chk["valid"]: errs.append(f"kkt_invalid:{chk[\'reason\']}")\n'
            '    elif "weights" in out and float(np.max(np.abs(np.asarray(out["weights"], dtype=float) - np.asarray(out["kkt_certificate"]["weights"], dtype=float)))) > 1e-5: errs.append("weights_cert_mismatch")\n'
        )
        priv_body = f'''
t0 = time.monotonic()
valid = True; max_w_rel = 0.0; max_fa_rel = 0.0; max_fb_rel = 0.0; max_gap = 0.0; rows = []
for sd, st, gt in cases:
    out = cand_mod.{entry_name}(st)
    if not isinstance(out, dict) or not {{"weights", "terminal_field_channel_a", "terminal_field_channel_b", "kkt_certificate"}} <= set(out.keys()):
        valid = False; rows.append({{"seed": sd, "error": "missing_keys"}}); continue
    w = np.asarray(out["weights"], dtype=float)
    fa = np.asarray(out["terminal_field_channel_a"], dtype=float)
    fb = np.asarray(out["terminal_field_channel_b"], dtype=float)
    if w.shape != gt["w_opt"].shape or fa.shape != gt["terminal_field_A"].shape or fb.shape != gt["terminal_field_B"].shape:
        valid = False; rows.append({{"seed": sd, "error": "shape_mismatch"}}); continue
    if not (np.all(np.isfinite(w)) and np.all(np.isfinite(fa)) and np.all(np.isfinite(fb))):
        valid = False; rows.append({{"seed": sd, "error": "non_finite"}}); continue
    chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], st["total_budget"], out["kkt_certificate"], gap_tol=2e-4)
    if not chk["valid"] or float(np.max(np.abs(w - np.asarray(out["kkt_certificate"]["weights"], dtype=float)))) > 1e-5:
        valid = False; rows.append({{"seed": sd, "error": f"kkt:{{chk.get('reason')}}"}}); continue
    wr = float(np.linalg.norm(w - gt["w_opt"]) / np.linalg.norm(gt["w_opt"]))
    far = float(np.linalg.norm(fa - gt["terminal_field_A"]) / np.linalg.norm(gt["terminal_field_A"]))
    fbr = float(np.linalg.norm(fb - gt["terminal_field_B"]) / np.linalg.norm(gt["terminal_field_B"]))
    max_w_rel = max(max_w_rel, wr); max_fa_rel = max(max_fa_rel, far); max_fb_rel = max(max_fb_rel, fb_rel if False else fbr); max_gap = max(max_gap, float(chk["gap"]))
    rows.append({{"seed": sd, "w_rel_l2": round(wr, 8), "fa_rel_l2": round(far, 8), "fb_rel_l2": round(fbr, 8), "kkt_gap": round(float(chk["gap"]), 10)}})
el = time.monotonic() - t0
sp = bool(valid and len(rows) == 2 and max_w_rel <= 2e-4 and max_fa_rel <= 2e-4 and max_fb_rel <= 2e-4 and max_gap <= 1e-5 and el <= 18.0)
print(json.dumps({{"strict_pass": sp, "reward": 1.0 if sp else 0.0, "max_weights_rel_l2": round(max_w_rel, 8), "max_field_a_rel_l2": round(max_fa_rel, 8), "max_field_b_rel_l2": round(max_fb_rel, 8), "max_kkt_gap": round(max_gap, 10), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}}, allow_nan=False))
'''

    return HoldoutOODTask(
        task_id=task_id,
        short_name=short_name,
        wave=wave_label,
        scientific_domain="Multi-Bank Tracer Reaction-Transport & Certified Box-Simplex Tomography Stream",
        wormhole_family="fam_kronecker_sylvester_and_kkt_certification",
        declared_artifacts=(f"/holdout_task/{artifact_name}",),
        declared_callable_entrypoints=(entry_name,),
        instruction_text=instruction_text,
        public_setup_py=(
            "import json, math, os, pickle, sys\nimport numpy as np\n"
            + _M94_BETA_GEN_PY
            + f"\npub_st, pub_gt = {pub_call_expr}\n"
            f'with open("/holdout_task/{pkl_name}", "wb") as fh:\n'
            '    pickle.dump([{"study": pub_st, "gt": pub_gt}], fh)\n'
            'open("/holdout_task/validate_public_schema.py", "w").write("import importlib.util, json, math, os, pickle, sys\\nimport numpy as np\\n" + '
            + repr(_SHARED_BOX_SIMPLEX_KKT_CHECKER_PY)
            + ' + """\n'
            "_d = os.path.dirname(os.path.abspath(__file__))\n"
            f'cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "{artifact_name}")\n'
            'spec = importlib.util.spec_from_file_location("_cand", cand_p)\n'
            "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
            f'item = pickle.load(open(os.path.join(_d, "{pkl_name}"), "rb"))[0]\n'
            'st, gt = item["study"], item["gt"]\n'
            f"out = mod.{entry_name}(st)\n"
            'M, nx, ny = st["source_modes"].shape\n'
            "errs = []\n"
            'if not isinstance(out, dict): errs.append("not_dict")\n'
            "else:\n"
            + val_check
            + 'print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))\n'
            "sys.exit(0 if len(errs) == 0 else 1)\n"
            '""")\n'
        ),
        private_verifier_py=(
            "import json, math, os, sys, time\nimport numpy as np\nsys.path.insert(0, '/holdout_task')\n"
            f"import {mod_stem} as cand_mod\n"
            + _M94_BETA_GEN_PY
            + f"\ncases = [\n{priv_cases_py}\n]\n"
            + priv_body
        ),
    )


_ALPHA_COMMON_CAL_NOTE = (
    "1. FACILITY LOAD-CELL & REFERENCE-COUPON CALIBRATION:\n"
    "   - `dataset['calibration_load_cells']` contains `{'lab_id': (40,), 'cmd_stress': (40,), 'true_stress': (40,)}`: estimate each lab's\n"
    "     linear load-cell law `true_stress = stress_gain[c] * cmd_stress + stress_bias[c]` (`c in 0..3`) and convert BOTH `past_stress_cmd`\n"
    "     and `stress_hold_cmd` into physical stress `past_stress` and `stress_hold_0`!\n"
    "   - Concatenate `train` + `test` panels (`lab_id`, `stratum` `0..2`, `coupon_features_raw` `(N, 4)`, `ref_coupons` `(N, 20)`) to estimate\n"
    "     per-lab coupon `offset[c]` (median column mean of `ref_coupons`) and `gain[c]` (median `ddof=0` std), standardize `(X_raw - offset)/gain`,\n"
    "     and impute `NaN`s with `(lab, stratum)` cell-balanced column means -> `X_aligned` (`(N, 4)`).\n"
)

_BETA_COMMON_CAL_NOTE = (
    "1. SENSOR-BANK CALIBRATION & 1D CELL-CENTERED ADVECTION-DIFFUSION-REACTION-ROBIN STENCIL:\n"
    "   - `study['sensor_calibration_sweeps']` gives `{'bank_id': (48,), 'ref_signal': (48,), 'raw_readout': (48,)}` obeying\n"
    "     `raw_readout = bank_gain[b] * ref_signal + bank_offset[b]` (`b in 0..3`); invert using `study['row_bank_id']` to map\n"
    "     `study['y_obs_raw']` -> calibrated sensor vector `y_obs = (y_obs_raw - bank_offset[row_bank_id]) / bank_gain[row_bank_id]`!\n"
    "   - Build 1D operators `A_x` (`(nx, nx)` from `Lx, nx, kappa_x, vel_x, robin_x` plus `+ np.diag(study['reaction_x'])`) and `B_y` (`(ny, ny)` from `Ly, ny, kappa_y, vel_y, robin_y` plus `+ np.diag(study['reaction_y'])`)\n"
    "     with harmonic interior diffusion `(2*k_i*k_{i+1}/(k_i+k_{i+1}))/h^2`, Robin boundary `(rg*k_b)/(h*(rg*0.5*h+k_b))` at `0, n-1`,\n"
    "     and first-order upwind advection (`vel[i] >= 0`: `+v/h` diag, `-v/h` at `i-1`; `vel[i] < 0`: `+|v|/h` diag, `-|v|/h` at `i+1`).\n"
)


M94_CHAIN_ALPHA_SOURCE = _make_alpha_task(
    task_id="alpha_s00__polymer_coupon_creep_baseline",
    short_name="alpha_s00",
    wave_label="m94_source",
    artifact_name="alpha_s00_solver.py",
    entry_name="solve_alpha_s00",
    instruction_text=(
        "Implement `/holdout_task/alpha_s00_solver.py` exposing `def solve_alpha_s00(dataset: dict) -> dict:`\n\n"
        + _ALPHA_COMMON_CAL_NOTE
        + "2. VARIABLE-STEP 2-BRANCH HEREDITARY MEMORY (`z_m(0) = 0`, `0.20 <= tau_1 < tau_2 <= 8.0`, ~8% Cauchy glitches on `strain_curves_obs`):\n"
        "   - Propagate `z_m <- exp(-past_dt[s]/tau_m)*z_m + (1 - exp(-past_dt[s]/tau_m))*past_stress[:, s]` (`s = 0..H-1`).\n"
        "   - Hold curve: `eps(t_k) = strain_0 + theta_0*(stress_hold_0 - z_1)*(1 - exp(-t_k/tau_1)) + theta_1*(stress_hold_0 - z_2)*(1 - exp(-t_k/tau_2)) + (theta_2 + X_aligned @ theta_{3:7})*t_k`,\n"
        "     with `instant_rate_obs` measuring `dot{eps}(0+) = theta_0*(stress_hold_0 - z_1)/tau_1 + theta_1*(stress_hold_0 - z_2)/tau_2 + theta_2 + X_aligned @ theta_{3:7}`.\n"
        "3. RETURN `{'estimated_taus': (2,), 'hereditary_weights': (7,), 'predicted_strain': (N_test, K), 'initial_rate_pred': (N_test,)}`.\n"
        "- Public dataset: `/holdout_task/public_alpha_s00.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
    ),
    pub_call_expr="_gen_m94_alpha_cohort(9401, (0.58, 3.95), [0.69, 0.55, 0.09, 0.16, -0.13, 0.11, -0.08])",
    priv_cases_py=(
        "    (99701, *_gen_m94_alpha_cohort(99701, (0.54, 3.75), [0.71, 0.56, 0.08, 0.17, -0.14, 0.12, -0.08])),\n"
        "    (99702, *_gen_m94_alpha_cohort(99702, (0.67, 4.40), [0.64, 0.58, 0.10, -0.15, 0.13, -0.11, 0.09])),"
    ),
    mode="standard",
)

M94_CHAIN_ALPHA_STREAM: tuple[HoldoutOODTask, ...] = (
    _make_alpha_task(
        task_id="alpha_ep01__preloaded_coupon_compliance_batch",
        short_name="alpha_ep01",
        wave_label="m94_stream_step_1",
        artifact_name="alpha_ep01_solver.py",
        entry_name="solve_alpha_ep01",
        instruction_text=(
            "Implement `/holdout_task/alpha_ep01_solver.py` exposing `def solve_alpha_ep01(dataset: dict) -> dict:`\n\n"
            + _ALPHA_COMMON_CAL_NOTE
            + "2. PRE-LOADED INITIAL MEMORY + SHIFTED COMPLIANCE & CROSSING READOUTS (`H = 22`, `K = 12`):\n"
            "   - Initialize `z_m` from `initial_memory_states` `(N, 2)` before propagating across `past_dt` with calibrated `past_stress`.\n"
            "   - Return `{'estimated_taus': (2,), 'predicted_strain': (N_test, K), 'secant_compliance': (predicted_strain[:, -1] - test['strain_0']) / np.maximum(np.abs(stress_hold_0), 0.5), 'hazard_crossing_times': (N_test,)}`\n"
            "     where `hazard_crossing_times` is the earliest piecewise-linear crossing of `(0, strain_0) -> (eval_times, predicted_strain)` at `test['critical_strain']` (`0.0` if `strain_0 >= crit`, `eval_times[-1]` if never reached).\n"
            "- Public dataset: `/holdout_task/public_alpha_ep01.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
        ),
        pub_call_expr='_gen_m94_alpha_cohort(9411, (0.45, 3.30), [0.74, 0.52, 0.08, 0.15, -0.12, 0.10, -0.08], H=22, K=12, dt_mode="geom_var", preload_scale=2.15)',
        priv_cases_py=(
            '    (99711, *_gen_m94_alpha_cohort(99711, (0.43, 3.20), [0.75, 0.51, 0.09, -0.16, 0.13, -0.10, 0.08], H=22, K=12, dt_mode="geom_var", preload_scale=2.15)),\n'
            '    (99712, *_gen_m94_alpha_cohort(99712, (0.76, 5.05), [0.62, 0.60, 0.07, 0.14, -0.15, 0.11, -0.09], H=22, K=12, dt_mode="geom_var", preload_scale=2.15)),'
        ),
        mode="transfer_T",
    ),
    _make_alpha_task(
        task_id="alpha_ep02__facility_recalibrated_coupon_batch",
        short_name="alpha_ep02",
        wave_label="m94_stream_step_2",
        artifact_name="alpha_ep02_solver.py",
        entry_name="solve_alpha_ep02",
        instruction_text=(
            "Implement `/holdout_task/alpha_ep02_solver.py` exposing `def solve_alpha_ep02(dataset: dict) -> dict:`\n\n"
            + _ALPHA_COMMON_CAL_NOTE
            + "2. HEREDITARY SPECTRUM & CREEP FORECASTING:\n"
            "   - Using the calibrated physical stresses `past_stress` and `stress_hold_0` (`z_m(0) = 0`, ~8% Cauchy glitches on `strain_curves_obs`),\n"
            "     identify `(tau_1, tau_2)` (`0.20 <= tau_1 < tau_2 <= 8.0`) and `hereditary_weights` `(7,)`.\n"
            "   - Return `{'estimated_taus': (2,), 'hereditary_weights': (7,), 'predicted_strain': (N_test, K), 'initial_rate_pred': (N_test,)}`.\n"
            "- Public dataset: `/holdout_task/public_alpha_ep02.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
        ),
        pub_call_expr=f"_gen_m94_alpha_cohort(9421, (0.52, 3.70), [0.70, 0.54, 0.08, -0.15, 0.13, 0.11, -0.09], stress_gains={ALPHA_G_GAINS}, stress_biases={ALPHA_G_BIASES})",
        priv_cases_py=(
            f"    (99721, *_gen_m94_alpha_cohort(99721, (0.49, 3.60), [0.72, 0.55, 0.09, 0.16, -0.13, 0.11, -0.08], stress_gains={ALPHA_G_GAINS}, stress_biases={ALPHA_G_BIASES})),\n"
            f"    (99722, *_gen_m94_alpha_cohort(99722, (0.69, 4.55), [0.63, 0.59, 0.08, -0.14, 0.14, -0.10, 0.09], stress_gains={ALPHA_G_GAINS}, stress_biases={ALPHA_G_BIASES})),"
        ),
        mode="standard",
    ),
    _make_alpha_task(
        task_id="alpha_ep03__high_load_structural_coupon_batch",
        short_name="alpha_ep03",
        wave_label="m94_stream_step_3",
        artifact_name="alpha_ep03_solver.py",
        entry_name="solve_alpha_ep03",
        instruction_text=(
            "Implement `/holdout_task/alpha_ep03_solver.py` exposing `def solve_alpha_ep03(dataset: dict) -> dict:`\n\n"
            + _ALPHA_COMMON_CAL_NOTE
            + "2. HIGH-LOAD STRUCTURAL REGIME (`dataset['yield_stress'] = sigma_y`):\n"
            "   - Calibrate physical `past_stress` and `stress_hold_0` from `calibration_load_cells` first!\n"
            "   - Below `sigma_y` (`|past_stress| <= sigma_y` and `|stress_hold_0| <= sigma_y`), the 2-branch linear hereditary law holds.\n"
            "   - Whenever calibrated stress exceeds `sigma_y`, quadratic hold overstress `ov_hold = (maximum(0, |stress_hold_0| - sigma_y)^2) * sign(stress_hold_0)`\n"
            "     adds `+ theta_7 * ov_hold * t_k`, and plastic-work history `D_hist = sum((maximum(0, |past_stress| - sigma_y)^2) * past_dt, axis=1)`\n"
            "     adds `+ theta_8 * (D_hist * stress_hold_0) * (1 - exp(-t_k / tau_1))` (and matching terms in `instant_rate_obs` at `t=0+`).\n"
            "   - Return `{'estimated_taus': (2,), 'viscoplastic_weights': [theta_7, theta_8] (2,), 'predicted_strain': (N_test, K)}`.\n"
            "- Public dataset: `/holdout_task/public_alpha_ep03.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
        ),
        pub_call_expr=f"_gen_m94_alpha_cohort(9431, (0.56, 3.85), [0.68, 0.54, 0.08, 0.15, -0.12, 0.11, -0.09], stress_gains={ALPHA_G_GAINS}, stress_biases={ALPHA_G_BIASES}, yield_stress=1.55, vp_theta=(0.24, 0.16))",
        priv_cases_py=(
            f"    (99731, *_gen_m94_alpha_cohort(99731, (0.51, 3.65), [0.70, 0.55, 0.09, 0.16, -0.13, 0.10, -0.08], stress_gains={ALPHA_G_GAINS}, stress_biases={ALPHA_G_BIASES}, yield_stress=1.50, vp_theta=(0.26, 0.18))),\n"
            f"    (99732, *_gen_m94_alpha_cohort(99732, (0.67, 4.40), [0.65, 0.57, 0.08, -0.14, 0.12, -0.11, 0.09], stress_gains={ALPHA_G_GAINS}, stress_biases={ALPHA_G_BIASES}, yield_stress=1.60, vp_theta=(0.22, 0.15))),"
        ),
        mode="counterexample_C",
    ),
    _make_alpha_task(
        task_id="alpha_ep04__standard_load_fresh_coupon_batch",
        short_name="alpha_ep04",
        wave_label="m94_stream_step_4",
        artifact_name="alpha_ep04_solver.py",
        entry_name="solve_alpha_ep04",
        instruction_text=(
            "Implement `/holdout_task/alpha_ep04_solver.py` exposing `def solve_alpha_ep04(dataset: dict) -> dict:`\n\n"
            + _ALPHA_COMMON_CAL_NOTE
            + "2. FRESH SPECIMEN BATCH (`dataset['specimen_reset'] == True`, `test['residual_plastic_state'] == 0`, `dataset['yield_stress'] = 2.85`):\n"
            "   - Calibrate physical `past_stress` and `stress_hold_0` from `calibration_load_cells` and inspect whether stresses exceed `yield_stress`.\n"
            "   - Identify `(tau_1, tau_2)`, `hereditary_weights` `(7,)`, `predicted_strain` `(N_test, K)`, and `initial_rate_pred` `(N_test,)`.\n"
            "- Public dataset: `/holdout_task/public_alpha_ep04.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
        ),
        pub_call_expr=f"_gen_m94_alpha_cohort(9441, (0.50, 3.55), [0.71, 0.53, 0.08, -0.15, 0.12, 0.10, -0.09], stress_gains={ALPHA_G_GAINS}, stress_biases={ALPHA_G_BIASES}, yield_stress=2.85, vp_theta=(0.0, 0.0), include_reset_metadata=True)",
        priv_cases_py=(
            f"    (99741, *_gen_m94_alpha_cohort(99741, (0.47, 3.50), [0.73, 0.55, 0.09, 0.16, -0.13, 0.11, -0.08], stress_gains={ALPHA_G_GAINS}, stress_biases={ALPHA_G_BIASES}, yield_stress=2.85, vp_theta=(0.0, 0.0), include_reset_metadata=True)),\n"
            f"    (99742, *_gen_m94_alpha_cohort(99742, (0.72, 4.65), [0.63, 0.59, 0.08, -0.14, 0.14, -0.10, 0.09], stress_gains={ALPHA_G_GAINS}, stress_biases={ALPHA_G_BIASES}, yield_stress=2.85, vp_theta=(0.0, 0.0), include_reset_metadata=True)),"
        ),
        mode="standard",
    ),
)


M94_CHAIN_BETA_SOURCE = _make_beta_task(
    task_id="beta_s00__tracer_transport_tomography_baseline",
    short_name="beta_s00",
    wave_label="m94_source",
    artifact_name="beta_s00_solver.py",
    entry_name="solve_beta_s00",
    instruction_text=(
        "Implement `/holdout_task/beta_s00_solver.py` exposing `def solve_beta_s00(study: dict) -> dict:`\n\n"
        + _BETA_COMMON_CAL_NOTE
        + "2. MULTI-SHIFT SYLVESTER TRANSPORT & CERTIFIED BOX-SIMPLEX KKT (`44 x 40`, `M = 14`, `K_mu = 10`):\n"
        "   - Solve `A_x @ U_{m,k} + U_{m,k} @ B_y.T + mu_shifts[k]*U_{m,k} = source_modes[m]`, stack sensor readouts `U_{:, k, si, sj}.T` (`k=0..K_mu-1`),\n"
        "     profile out `Z_nuisance` (`ridge = study['nuisance_ridge']`) against calibrated `y_obs`, and minimize\n"
        "     `0.5*||S_tilde @ w - r_y||_2^2 + 0.5*source_ridge*||w||_2^2` on `{0 <= w <= u_box, sum(w) == total_budget}`.\n"
        "   - Return `{'weights': (M,), 'terminal_field': sum_m w_m * U_{m, K_mu-1} (nx, ny), 'kkt_certificate': {'weights', 'nu_sum', 'lam_lower', 'lam_upper', 'primal_objective', 'dual_lower_bound'}}`.\n"
        "- Public study: `/holdout_task/public_beta_s00.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
    ),
    pub_call_expr="_gen_m94_beta_study(9501)",
    priv_cases_py=(
        "    (99801, *_gen_m94_beta_study(99801)),\n"
        "    (99802, *_gen_m94_beta_study(99802)),"
    ),
    mode="standard",
)

M94_CHAIN_BETA_STREAM: tuple[HoldoutOODTask, ...] = (
    _make_beta_task(
        task_id="beta_ep01__anisotropic_multibudget_survey_batch",
        short_name="beta_ep01",
        wave_label="m94_stream_step_1",
        artifact_name="beta_ep01_solver.py",
        entry_name="solve_beta_ep01",
        instruction_text=(
            "Implement `/holdout_task/beta_ep01_solver.py` exposing `def solve_beta_ep01(study: dict) -> dict:`\n\n"
            + _BETA_COMMON_CAL_NOTE
            + "2. ANISOTROPIC GRID (`48 x 36`, `M = 16`, `K_mu = 12`) & MULTI-BUDGET KKT PARETO (`budget_grid` of length `3`):\n"
            "   - Calibrate `y_obs` from `y_obs_raw`, profile `Z_nuisance`, and solve the Box-Simplex QP + KKT dual certificate for each `B_b` in `budget_grid`.\n"
            "   - Return `{'weights_by_budget': (3, M), 'terminal_field_mid_budget': (nx, ny) for b=1, 'kkt_certificates': [cert_0, cert_1, cert_2]}`.\n"
            "- Public study: `/holdout_task/public_beta_ep01.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
        ),
        pub_call_expr="_gen_m94_beta_study(9511, nx=48, ny=36, K_mu=12, M=16, vel_sign=-1.15, robin_pair=(0.75, 2.05), multi_budget=True)",
        priv_cases_py=(
            "    (99811, *_gen_m94_beta_study(99811, nx=48, ny=36, K_mu=12, M=16, vel_sign=-1.15, robin_pair=(0.75, 2.05), multi_budget=True)),\n"
            "    (99812, *_gen_m94_beta_study(99812, nx=48, ny=36, K_mu=12, M=16, vel_sign=-1.15, robin_pair=(0.75, 2.05), multi_budget=True)),"
        ),
        mode="multibudget_T",
    ),
    _make_beta_task(
        task_id="beta_ep02__recalibrated_sensor_array_survey_batch",
        short_name="beta_ep02",
        wave_label="m94_stream_step_2",
        artifact_name="beta_ep02_solver.py",
        entry_name="solve_beta_ep02",
        instruction_text=(
            "Implement `/holdout_task/beta_ep02_solver.py` exposing `def solve_beta_ep02(study: dict) -> dict:`\n\n"
            + _BETA_COMMON_CAL_NOTE
            + "2. SEPARABLE MULTI-SHIFT TRANSPORT & BOX-SIMPLEX KKT:\n"
            "   - Calibrate `y_obs` from `y_obs_raw` via `sensor_calibration_sweeps` + `row_bank_id`, solve separable `44 x 40` Sylvester shifts,\n"
            "     profile `Z_nuisance`, and return `{'weights': (M,), 'terminal_field': (nx, ny), 'kkt_certificate': dict}`.\n"
            "- Public study: `/holdout_task/public_beta_ep02.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
        ),
        pub_call_expr=f"_gen_m94_beta_study(9521, bank_gains={BETA_G_GAINS}, bank_offsets={BETA_G_OFFS})",
        priv_cases_py=(
            f"    (99821, *_gen_m94_beta_study(99821, bank_gains={BETA_G_GAINS}, bank_offsets={BETA_G_OFFS})),\n"
            f"    (99822, *_gen_m94_beta_study(99822, bank_gains={BETA_G_GAINS}, bank_offsets={BETA_G_OFFS})),"
        ),
        mode="standard",
    ),
    _make_beta_task(
        task_id="beta_ep03__dual_tracer_reactive_survey_batch",
        short_name="beta_ep03",
        wave_label="m94_stream_step_3",
        artifact_name="beta_ep03_solver.py",
        entry_name="solve_beta_ep03",
        instruction_text=(
            "Implement `/holdout_task/beta_ep03_solver.py` exposing `def solve_beta_ep03(study: dict) -> dict:`\n\n"
            + _BETA_COMMON_CAL_NOTE
            + "2. DUAL-TRACER REACTIVE SURVEY (`36 x 32`, `M = 12`, `K_mu = 8`):\n"
            "   - Calibrate `y_obs` (`2 * K_mu * N_sens` rows) from `y_obs_raw` using `sensor_calibration_sweeps` + `row_bank_id`!\n"
            "   - **Channel A**: `A_x @ U^A + U^A @ B_y.T + (V_channel_a + mu_k)*U^A = source_modes[m]` where `V_channel_a(x,y) = v_x(x) + v_y(y)` is\n"
            "     purely additive separable (absorbable into 1D diagonals of `A_x` and `B_y`!).\n"
            "   - **Channel B**: `A_x @ U^B + U^B @ B_y.T + (V_channel_b + mu_k)*U^B = source_modes[m]` where `V_channel_b(x,y)` has strong\n"
            "     non-separable 2D cross-coupling (`kron(Ax, I_y) + kron(I_x, By) + diag(V_channel_b.ravel() + mu_k)` in C-order).\n"
            "   - Stack `[S_A; S_B]`, profile `Z_nuisance`, and return `{'weights': (M,), 'terminal_field_channel_a': (nx, ny), 'terminal_field_channel_b': (nx, ny), 'kkt_certificate': dict}`.\n"
            "- Public study: `/holdout_task/public_beta_ep03.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
        ),
        pub_call_expr=f"_gen_m94_beta_study(9531, nx=36, ny=32, K_mu=8, M=12, bank_gains={BETA_G_GAINS}, bank_offsets={BETA_G_OFFS}, dual_reactive_channels=True)",
        priv_cases_py=(
            f"    (99831, *_gen_m94_beta_study(99831, nx=36, ny=32, K_mu=8, M=12, bank_gains={BETA_G_GAINS}, bank_offsets={BETA_G_OFFS}, dual_reactive_channels=True)),\n"
            f"    (99832, *_gen_m94_beta_study(99832, nx=36, ny=32, K_mu=8, M=12, bank_gains={BETA_G_GAINS}, bank_offsets={BETA_G_OFFS}, dual_reactive_channels=True)),"
        ),
        mode="dual_C",
    ),
    _make_beta_task(
        task_id="beta_ep04__single_tracer_background_survey_batch",
        short_name="beta_ep04",
        wave_label="m94_stream_step_4",
        artifact_name="beta_ep04_solver.py",
        entry_name="solve_beta_ep04",
        instruction_text=(
            "Implement `/holdout_task/beta_ep04_solver.py` exposing `def solve_beta_ep04(study: dict) -> dict:`\n\n"
            + _BETA_COMMON_CAL_NOTE
            + "2. SINGLE-TRACER SURVEY WITH BACKGROUND POTENTIAL `study['V_background']` (`44 x 40`, `M = 14`, `K_mu = 10`):\n"
            "   - Calibrate `y_obs` from `y_obs_raw` using `sensor_calibration_sweeps` + `row_bank_id`!\n"
            "   - Solve `A_x @ U_{m,k} + U_{m,k} @ B_y.T + (V_background + mu_shifts[k]) * U_{m,k} = source_modes[m]` (inspect whether\n"
            "     `V_background(x, y)` is additively separable `v_x(x) + v_y(y)` or non-separable!), profile `Z_nuisance`, and return\n"
            "     `{'weights': (M,), 'terminal_field': (nx, ny), 'kkt_certificate': dict}`.\n"
            "- Public study: `/holdout_task/public_beta_ep04.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
        ),
        pub_call_expr=f"_gen_m94_beta_study(9541, bank_gains={BETA_G_GAINS}, bank_offsets={BETA_G_OFFS}, additive_background=True)",
        priv_cases_py=(
            f"    (99841, *_gen_m94_beta_study(99841, bank_gains={BETA_G_GAINS}, bank_offsets={BETA_G_OFFS}, additive_background=True)),\n"
            f"    (99842, *_gen_m94_beta_study(99842, bank_gains={BETA_G_GAINS}, bank_offsets={BETA_G_OFFS}, additive_background=True)),"
        ),
        mode="standard",
    ),
)

M94_ALL_TASKS: tuple[HoldoutOODTask, ...] = (
    M94_CHAIN_ALPHA_SOURCE,
    *M94_CHAIN_ALPHA_STREAM,
    M94_CHAIN_BETA_SOURCE,
    *M94_CHAIN_BETA_STREAM,
)
M94_TASKS_BY_ID: dict[str, HoldoutOODTask] = {t.task_id: t for t in M94_ALL_TASKS}


def grade_m94_artifact_map(task: HoldoutOODTask, files_map: dict[str, str]) -> dict[str, Any]:
    """Package `files_map` (`{'/holdout_task/...': code}`) into an ephemeral POSIX `.tar` and run `run_holdout_private_verifier`."""
    import io
    import tarfile
    import tempfile

    with tempfile.TemporaryDirectory(prefix="m94_tar_") as td:
        tar_p = Path(td) / "submitted.tar"
        with tarfile.open(tar_p, mode="w") as tf:
            for abs_p, src_txt in sorted(files_map.items()):
                raw = str(src_txt).encode("utf-8")
                ti = tarfile.TarInfo(name=str(abs_p).lstrip("/"))
                ti.size = len(raw)
                ti.mode = 0o644
                tf.addfile(ti, io.BytesIO(raw))
        return run_holdout_private_verifier(task, tar_p)


def m94_task_to_frontier_task(
    task: HoldoutOODTask,
    *,
    continual_context_block: str = "",
) -> m6f.FrontierBenchmarkTask:
    """Wrap an `M9.4` task into `FrontierBenchmarkTask` with zero-leakage neutral task metadata."""
    full_inst = (
        f"{continual_context_block.strip()}\n\n{task.instruction_text}"
        if continual_context_block.strip()
        else task.instruction_text
    )
    pkl_asset = f"/holdout_task/public_{task.short_name}.pkl"
    return m6f.FrontierBenchmarkTask(
        task_id=task.task_id,
        short_name=task.short_name,
        benchmark="Hegel-M94-Continual-Binding-World",
        wave=task.wave,
        scientific_domain=task.scientific_domain,
        wormhole_family=task.wormhole_family,
        instruction_text=full_inst,
        declared_artifacts=task.declared_artifacts,
        declared_callable_entrypoints=task.declared_callable_entrypoints,
        agent_image_tag=CLEAN_SCIPY_SANDBOX_IMAGE,
        verifier_image_tag=CLEAN_SCIPY_SANDBOX_IMAGE,
        agent_cpus=2,
        agent_memory_mb=3072,
        verifier_cpus=2,
        verifier_memory_mb=2048,
        verifier_timeout_sec=120,
        verifier_network_mode="none",
        holdout_public_setup_py=task.public_setup_py,
        required_public_assets=(pkl_asset, "/holdout_task/validate_public_schema.py"),
    )


def render_bound_executable_adapter(
    *,
    binding: MechanismExecutableBinding,
    target_entrypoint: str,
    param_payload: dict[str, Any],
    graph: PersistentMechanismGraph | None = None,
) -> str:
    """Bind the ACTUAL agent-synthesized Python module (`binding.parameterized_closure_code` from `Stage S` or `C1`)
    to the current `PersistentMechanismGraph` `param_payload` (`AncestorReweight` epoch) via a pure input-role adapter.

    CRITICAL AUDIT INVARIANT (`M9.4.1` `Gate B`):
      1. Zero numerical solver logic is emitted here — 100% of scientific computation runs inside `binding.parameterized_closure_code`!
      2. Strictly verifies declared SHA-256, qualification_id, and (when `graph` is provided) foreign keys & parameter epoch BEFORE rendering!
    """
    import base64

    verify_binding_identity_and_foreign_keys(binding, graph=graph, require_closure_code=True)
    src_code = str(binding.parameterized_closure_code or "")
    src_ep = str(binding.parameterized_entrypoint or "")
    b64_src = base64.b64encode(src_code.encode("utf-8")).decode("ascii")
    src_sha = hashlib.sha256(src_code.encode("utf-8")).hexdigest()
    param_sha = hashlib.sha256(json.dumps(param_payload, sort_keys=True).encode("utf-8")).hexdigest()

    if binding.chain_id == "alpha":
        gains_lit = [round(float(x), 8) for x in param_payload["stress_gains"]]
        biases_lit = [round(float(x), 8) for x in param_payload["stress_biases"]]
        return (
            f'"""M9.4 MechanismExecutableBinding Adapter (`{binding.binding_id}` @ `{binding.parameter_mechanism_id}` v{binding.parameter_version}, '
            f'bound_source_sha256={src_sha[:16]})."""\n'
            "from __future__ import annotations\n"
            "import base64, copy, types\n"
            "import numpy as np\n\n"
            f"_BOUND_SOURCE_B64 = {b64_src!r}\n"
            f"_BOUND_SOURCE_ENTRYPOINT = {src_ep!r}\n"
            f"_BOUND_STRESS_GAINS = np.array({gains_lit!r}, dtype=float)\n"
            f"_BOUND_STRESS_BIASES = np.array({biases_lit!r}, dtype=float)\n\n"
            "_bound_mod = types.ModuleType('_hegel_m94_bound_source_alpha')\n"
            "exec(compile(base64.b64decode(_BOUND_SOURCE_B64).decode('utf-8'), '<bound_stage_s_alpha>', 'exec'), _bound_mod.__dict__)\n"
            "_bound_fn = getattr(_bound_mod, _BOUND_SOURCE_ENTRYPOINT)\n\n"
            f"def {target_entrypoint}(dataset: dict) -> dict:\n"
            "    ds2 = copy.deepcopy(dataset)\n"
            "    for sp_k in ('train', 'test'):\n"
            "        sp = dict(ds2[sp_k])\n"
            "        lb = np.asarray(sp['lab_id'], dtype=int).ravel()\n"
            "        sp['past_stress_cmd'] = np.asarray(sp['past_stress_cmd'], dtype=float) * _BOUND_STRESS_GAINS[lb, None] + _BOUND_STRESS_BIASES[lb, None]\n"
            "        sp['stress_hold_cmd'] = np.asarray(sp['stress_hold_cmd'], dtype=float) * _BOUND_STRESS_GAINS[lb] + _BOUND_STRESS_BIASES[lb]\n"
            "        ds2[sp_k] = sp\n"
            "    if 'calibration_load_cells' in ds2:\n"
            "        cal_d = dict(ds2['calibration_load_cells'])\n"
            "        cal_d['true_stress'] = np.asarray(cal_d['cmd_stress'], dtype=float).copy()\n"
            "        ds2['calibration_load_cells'] = cal_d\n"
            "    return _bound_fn(ds2)\n"
        )

    gains_b = [round(float(x), 8) for x in param_payload["bank_gains"]]
    offs_b = [round(float(x), 8) for x in param_payload["bank_offsets"]]
    return (
        f'"""M9.4 MechanismExecutableBinding Adapter (`{binding.binding_id}` @ `{binding.parameter_mechanism_id}` v{binding.parameter_version}, '
        f'bound_source_sha256={src_sha[:16]})."""\n'
        "from __future__ import annotations\n"
        "import base64, copy, types\n"
        "import numpy as np\n\n"
        f"_BOUND_SOURCE_B64 = {b64_src!r}\n"
        f"_BOUND_SOURCE_ENTRYPOINT = {src_ep!r}\n"
        f"_BOUND_BANK_GAINS = np.array({gains_b!r}, dtype=float)\n"
        f"_BOUND_BANK_OFFSETS = np.array({offs_b!r}, dtype=float)\n\n"
        "_bound_mod = types.ModuleType('_hegel_m94_bound_source_beta')\n"
        "exec(compile(base64.b64decode(_BOUND_SOURCE_B64).decode('utf-8'), '<bound_stage_s_beta>', 'exec'), _bound_mod.__dict__)\n"
        "_bound_fn = getattr(_bound_mod, _BOUND_SOURCE_ENTRYPOINT)\n\n"
        f"def {target_entrypoint}(study: dict) -> dict:\n"
        "    st2 = copy.deepcopy(study)\n"
        "    raw_y = np.asarray(st2['y_obs_raw'], dtype=float).ravel()\n"
        "    rb = np.asarray(st2['row_bank_id'], dtype=int).ravel()\n"
        "    st2['y_obs_raw'] = (raw_y - _BOUND_BANK_OFFSETS[rb]) / _BOUND_BANK_GAINS[rb]\n"
        "    if 'sensor_calibration_sweeps' in st2:\n"
        "        cal_s = dict(st2['sensor_calibration_sweeps'])\n"
        "        cal_s['raw_readout'] = np.asarray(cal_s['ref_signal'], dtype=float).copy()\n"
        "        st2['sensor_calibration_sweeps'] = cal_s\n"
        "    if st2.get('V_background') is not None:\n"
        "        v_bg = np.asarray(st2.pop('V_background'), dtype=float)\n"
        "        nx_i, ny_i = int(st2['nx']), int(st2['ny'])\n"
        "        rx0 = np.asarray(st2.get('reaction_x', np.zeros(nx_i)), dtype=float)\n"
        "        ry0 = np.asarray(st2.get('reaction_y', np.zeros(ny_i)), dtype=float)\n"
        "        st2['reaction_x'] = rx0 + np.mean(v_bg, axis=1)\n"
        "        st2['reaction_y'] = ry0 + (np.mean(v_bg, axis=0) - float(np.mean(v_bg)))\n"
        "    return _bound_fn(st2)\n"
    )


def evaluate_bound_mechanism_candidate_in_container(
    *,
    task: HoldoutOODTask,
    chain_id: str,
    graph: PersistentMechanismGraph,
    registry: M94BindingRegistry,
    public_item: dict[str, Any],
) -> dict[str, Any]:
    """Uniform, task-ID-blind pre-LLM `MechanismExecutableBinding` evaluation (`M9.4` Core Continual Engine!).

    1. Evaluates `ObservablePreconditionGuard` (`TRUE / FALSE / UNKNOWN`) for every registered binding on `chain_id`
       against calibrated public observables (`0` private verifier peek!).
    2. Filters bindings whose guard is `TRUE` AND whose `output_schema_keys` match the public schema validator.
    3. Binds the ACTUAL agent-synthesized executable (`chosen_b.parameterized_closure_code`) to the current `shared_upstream`
       parameter epoch via `render_bound_executable_adapter` inside a clean air-gapped container and runs BOTH
       `/holdout_task/validate_public_schema.py` (`PublicSchemaReceipt`) AND `execute_host_trusted_scientific_probe` (`ScientificProbeReceipt`).
    4. Accepts direct bound execution ONLY when guard == `TRUE`, schema_valid == `True`, AND `ScientificProbeReceipt`
       verifies quantitative physical/KKT invariants! Otherwise returns structured component/guard diagnostics for LLM synthesis.
    """
    import subprocess
    import tempfile

    shared_id = (
        "alpha_shared_loadcell_calibration"
        if chain_id == "alpha"
        else "beta_shared_sensor_bank_calibration"
    )
    param_ep = registry.parameter_epochs[shared_id]
    param_payload = param_ep["payload"]
    obs_ctx = extract_guard_observables_from_m94_public_item(chain_id, public_item, param_payload)

    chain_bindings = [b for b in registry.bindings.values() if b.chain_id == chain_id]
    guard_evaluations: list[dict[str, Any]] = []
    eligible_bindings: list[MechanismExecutableBinding] = []

    # Inspect expected output keys ONLY from the `validate_public_schema.py` block (NOT generator code!)
    val_schema_txt = task.public_setup_py.split("validate_public_schema.py", 1)[-1]
    for b in chain_bindings:
        g_res = evaluate_observable_precondition_guard(b.precondition_guard, obs_ctx)
        keys_present_in_schema = all(f'"{k}"' in val_schema_txt for k in b.output_schema_keys) and (
            ("secant_compliance" not in val_schema_txt or "secant_compliance" in b.output_schema_keys)
            and ("viscoplastic_weights" not in val_schema_txt or "viscoplastic_weights" in b.output_schema_keys)
            and ("weights_by_budget" not in val_schema_txt or "weights_by_budget" in b.output_schema_keys)
            and ("terminal_field_channel_b" not in val_schema_txt or "terminal_field_channel_b" in b.output_schema_keys)
        )
        guard_evaluations.append(
            {
                "binding_id": b.binding_id,
                "branch_tag": b.branch_tag,
                "component_id": b.component_id,
                "parameter_version": b.parameter_version,
                "guard_result": g_res,
                "public_output_interface_compatible": keys_present_in_schema,
            }
        )
        if g_res.get("state") == "TRUE" and keys_present_in_schema:
            eligible_bindings.append(b)

    if not eligible_bindings:
        violated_notes = [
            ev for ev in guard_evaluations if ev["guard_result"].get("state") == "FALSE"
        ]
        return {
            "accepted_bound_execution": False,
            "reason": (
                "precondition_guard_rejected_existing_branches"
                if violated_notes and any(not ev["public_output_interface_compatible"] for ev in guard_evaluations)
                else ("interface_adaptation_required" if all(not ev["public_output_interface_compatible"] for ev in guard_evaluations) else "no_guard_matched_branch")
            ),
            "parameter_mechanism_id": shared_id,
            "parameter_version": int(param_ep["version"]),
            "parameter_payload": deepcopy(param_payload),
            "guard_evaluations": guard_evaluations,
        }

    chosen_b = eligible_bindings[0]
    entry_nm = task.declared_callable_entrypoints[0]
    art_path = task.declared_artifacts[0]
    cand_src = render_bound_executable_adapter(
        binding=chosen_b,
        target_entrypoint=entry_nm,
        param_payload=param_payload,
        graph=graph,
    )
    chosen_b.adapter_code_sha256 = hashlib.sha256(cand_src.encode("utf-8")).hexdigest()
    chosen_b.parameter_epoch_sha256 = hashlib.sha256(json.dumps(param_payload, sort_keys=True).encode("utf-8")).hexdigest()

    ft = m94_task_to_frontier_task(task)
    cid = m6f.start_detached_container(CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="2048m", cpus="2", network_mode="none")
    try:
        with tempfile.TemporaryDirectory(prefix="m94_btrial_") as td:
            td_p = Path(td)
            (td_p / "_setup.py").write_text(task.public_setup_py, encoding="utf-8")
            (td_p / "cand.py").write_text(cand_src, encoding="utf-8")
            m6f.exec_container_bash(cid, "mkdir -p /holdout_task", timeout_sec=10)
            subprocess.run(["docker", "cp", str(td_p / "_setup.py"), f"{cid}:/tmp/_setup.py"], check=True, timeout=15)
            m6f.exec_container_bash(cid, "python3 -I /tmp/_setup.py && rm -f /tmp/_setup.py", timeout_sec=45)
            # Record public setup SHA-256 map
            import shlex as _shlex_m94

            pub_shas_py = (
                "import hashlib, json, os\n"
                f"ps = {json.dumps(list(ft.required_public_assets))}\n"
                "print(json.dumps({p: hashlib.sha256(open(p,'rb').read()).hexdigest() for p in ps if os.path.isfile(p)}, sort_keys=True))\n"
            )
            sh_r = m6f.exec_container_bash(cid, "python3 -I -c " + _shlex_m94.quote(pub_shas_py), timeout_sec=15)
            pub_shas = json.loads(sh_r["stdout"].strip().splitlines()[-1])
            subprocess.run(["docker", "cp", str(td_p / "cand.py"), f"{cid}:{art_path}"], check=True, timeout=15)

        cand_bundle_sha = m6f.compute_declared_artifacts_digest(cid, ft.declared_artifacts)
        sch_r = m6f.exec_container_bash(
            cid,
            "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py",
            timeout_sec=45,
        )
        sch_ok = sch_r["exit_code"] == 0 and '"schema_valid": true' in sch_r["stdout"].lower()
        sci_rec = (
            m6f.execute_host_trusted_scientific_probe(
                agent_cid=cid,
                task=ft,
                current_candidate_sha256=cand_bundle_sha,
                public_setup_files_sha256=pub_shas,
            )
            if sch_ok
            else None
        )
        if sch_ok and isinstance(sci_rec, dict) and sci_rec.get("scientific_invariants_verified") is True:
            return {
                "accepted_bound_execution": True,
                "selected_binding_id": chosen_b.binding_id,
                "selected_branch_tag": chosen_b.branch_tag,
                "parameter_mechanism_id": shared_id,
                "parameter_version": int(param_ep["version"]),
                "parameter_payload": deepcopy(param_payload),
                "candidate_code": cand_src,
                "candidate_raw_sha256": hashlib.sha256(cand_src.encode("utf-8")).hexdigest(),
                "candidate_bundle_sha256": cand_bundle_sha,
                "guard_evaluations": guard_evaluations,
                "scientific_probe_receipt": sci_rec,
            }
        return {
            "accepted_bound_execution": False,
            "reason": "bound_closure_failed_public_scientific_probe",
            "parameter_mechanism_id": shared_id,
            "parameter_version": int(param_ep["version"]),
            "parameter_payload": deepcopy(param_payload),
            "guard_evaluations": guard_evaluations,
            "schema_valid": sch_ok,
            "scientific_probe_receipt": sci_rec,
        }
    finally:
        m6f.stop_container_safely(cid)
