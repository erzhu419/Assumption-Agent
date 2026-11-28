"""Hegel `M10.1`: Bounded Unknown-Cut Acyclic Computational DAG Discovery (`1–4` Nodes, `depth <= 4`, `fan-out <= 2`).

Eliminates `M10.0`'s prescribed two-stage split boundary (`calibrate_upstream -> solve_downstream`) while remaining
scientifically identifiable and falsifiable:
  1. **Zero Prescribed Split Boundaries / Module Counts / Role Labels**:
     - Every task across ALL structural topologies shares the EXACT same blind public input schema
       (`eval_times`, `panel_a`, `panel_b`, `panel_c`, `train`, `test`, `telemetry_log`) and output contract
       (`y_hat`, `state_pred`, `diagnostics_vec`) with neutral IDs (`node_00..`, `dag_00..`, `obs_00..obs_17`).
     - `K = 1` monolithic (`keep_monolithic`) is ALWAYS legal and zero-penalty when no reusable cut exists.
  2. **Bounded Acyclic Computational DAG Engine & 6 Legal Graph-Edit Actions**:
     - Bounds: `1 <= active_nodes <= 4`, acyclic (Kahn topological sort), `depth <= 4`, `max_fanout <= 2`.
     - 6 legal actions: `keep_monolithic`, `extract_module`, `split_node`, `merge_nodes`, `reuse_subgraph`, `rewire_edge`.
     - Runtime instrumentation in air-gapped Docker enforces `exec_count[nid] == 1` on every active node (proving
       shared diamond/skip prefixes execute ONCE and fan out deep-copied intermediate dicts without aliasing) and
       rejects trivial identity-shell nodes (`output_numeric_sha == input_numeric_sha`).
     - Full `ProbeReceiptStore` atomic persistence + 3-way external receipt FK on `CutWithdrawalMergeReceipt` (`T01`).
  3. **Four Blind Scientific Structural Contracts & Two-Stratum Confirmation Design (`n = 4` Dev + `n = 8` Confirmation)**:
     - Contract 1 (`monolithic_coupled`): Non-separable implicit nonlinear core (`K = 1` optimal; tests `keep_monolithic`).
     - Contract 2 (`chain_k3_variable_cut`, plus Stratum B held-out `chain_k4_deep_cascade`): Multi-stage cascades
       breaking fixed 2-stage chains (`K = 2`) with unseen 3-way / 4-way compositional shifts.
     - Contract 3 (`diamond_k4_shared_prefix`, plus Stratum B held-out `asymmetric_skip_dag_k4`): Shared prefix
       `fan-out = 2` feeding parallel branches that rejoin via bilinear conservation coupling (`exec_count == 1`).
     - Contract 4 (Cross-branch interaction falsifying an earlier cut -> `merge_nodes` / `rewire_edge` with selective
       guard expansion preserving the old separable DAG on colliding revisits; plus Stratum B held-out
       `diamond_k4_coupled_first` and `hybrid_mono_to_split_shift`).
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from time import monotonic
from typing import Any
import uuid

import numpy as np

from . import m6_frontier_real_benchmarks as m6f
from .m7_autogenous_forest_pretraining import CLEAN_SCIPY_SANDBOX_IMAGE
from .m7_holdout_ood_blind_benchmark import HoldoutOODTask
from .m8_persistent_variational_planner import PersistentMechanismGraph
from .m9_4_1_shared_verification_kernel import (
    SHARED_KERNEL_VERSION,
    compute_canonical_numeric_item_sha256,
)
from .m9_6_autonomous_structure_continual_world import (
    ProbeReceiptStore,
    _canonical_json_sha256,
    evaluate_m96_autonomous_guard,
)

M101_ARMS: tuple[str, ...] = (
    "Flat_MonolithicArchive_Online",
    "Fixed_TwoStage_Modular_Online",
    "Hegel_UnknownCut_GraphDiscovery_Online",
)

M101_LEGAL_DAG_ACTIONS: frozenset[str] = frozenset(
    {
        "keep_monolithic",
        "extract_module",
        "split_node",
        "merge_nodes",
        "reuse_subgraph",
        "rewire_edge",
    }
)

M101_OUTPUT_KEYS: tuple[str, ...] = ("y_hat", "state_pred", "diagnostics_vec")
M101_MAX_ACTIVE_NODES = 4
M101_MAX_DEPTH = 4
M101_MAX_FANOUT = 2

M101_OBSERVABLE_NAMES: tuple[str, ...] = (
    "obs_00_pa_lin_r2",
    "obs_01_pa_cross_corr",
    "obs_02_pa_cubic_gain",
    "obs_03_pa_rms",
    "obs_04_pb_lin_r2",
    "obs_05_pb_quad_gain",
    "obs_06_pb_harmonic_gain",
    "obs_07_pb_rms",
    "obs_08_pc_v1_lin_r2",
    "obs_09_pc_v1_nl_gain",
    "obs_10_pc_v2_lin_r2",
    "obs_11_pc_v2_nl_gain",
    "obs_12_tr_y_std",
    "obs_13_tr_traj_curv",
    "obs_14_tel_ch0",
    "obs_15_tel_ch1",
    "obs_16_tel_ch2",
    "obs_17_tel_ch3",
)

M101_DAG_LEDGER_FOREST_KEY = "__m10_1_unknown_cut_dag_state__"


# ============================================================================
# 1. Bounded Unknown-Cut Executable Computational DAG Data Structures & Compiler
# ============================================================================


@dataclass
class ExecutableModuleNode:
    """Self-contained executable Python computation node (`run_node(inputs: dict) -> dict`) with neutral IDs (`T09`)."""

    node_id: str
    version_id: str
    code: str
    code_sha256: str = ""
    applicability_guard: dict[str, Any] = field(default_factory=dict)
    priority: int = 10
    origin_action: str = "keep_monolithic"
    origin_step: int = 0
    probe_receipt_id: str = ""
    probe_receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.code_sha256:
            self.code_sha256 = hashlib.sha256(self.code.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "version_id": self.version_id,
            "code": self.code,
            "code_sha256": self.code_sha256,
            "applicability_guard": deepcopy(self.applicability_guard),
            "priority": int(self.priority),
            "origin_action": self.origin_action,
            "origin_step": int(self.origin_step),
            "probe_receipt_id": self.probe_receipt_id,
            "probe_receipt_sha256": self.probe_receipt_sha256,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExecutableModuleNode:
        return cls(
            node_id=str(d["node_id"]),
            version_id=str(d["version_id"]),
            code=str(d["code"]),
            code_sha256=str(d.get("code_sha256") or ""),
            applicability_guard=deepcopy(d.get("applicability_guard") or {}),
            priority=int(d.get("priority", 10)),
            origin_action=str(d.get("origin_action") or "keep_monolithic"),
            origin_step=int(d.get("origin_step", 0)),
            probe_receipt_id=str(d.get("probe_receipt_id") or ""),
            probe_receipt_sha256=str(d.get("probe_receipt_sha256") or ""),
        )


@dataclass
class ExecutableComputationDAG:
    """Bounded acyclic computational DAG (`1 <= K <= 4`, `depth <= 4`, `fan-out <= 2`, `K = 1` ALWAYS legal!)."""

    dag_id: str
    nodes: list[str]
    edges: list[tuple[str, str]]  # ("__input__", node_id) or (src_node_id, dst_node_id)
    sink_node_id: str
    node_versions: dict[str, str]  # node_id -> version_id
    applicability_guard: dict[str, Any] = field(default_factory=dict)
    priority: int = 10
    origin_action: str = "keep_monolithic"
    origin_step: int = 0
    is_cross_scoped: bool = False

    def validate_topology(self) -> dict[str, Any]:
        """Strictly validate bounded DAG invariants (`1 <= K <= 4`, acyclic, `depth <= 4`, `fan-out <= 2`, full reachability)."""
        n_set = set(self.nodes)
        if not (1 <= len(self.nodes) <= M101_MAX_ACTIVE_NODES) or len(n_set) != len(self.nodes):
            raise ValueError(f"DAG `{self.dag_id}` active node count {len(self.nodes)} outside [1, {M101_MAX_ACTIVE_NODES}]")
        if self.sink_node_id not in n_set:
            raise ValueError(f"DAG `{self.dag_id}` sink `{self.sink_node_id}` not in active nodes")
        if set(self.node_versions.keys()) != n_set:
            raise ValueError(f"DAG `{self.dag_id}` node_versions keys != active nodes")

        adj: dict[str, list[str]] = {n: [] for n in self.nodes}
        rev: dict[str, list[str]] = {n: [] for n in self.nodes}
        in_deg: dict[str, int] = {n: 0 for n in self.nodes}
        has_root_in: set[str] = set()
        seen_edges: set[tuple[str, str]] = set()

        for u, v in self.edges:
            if (u, v) in seen_edges:
                raise ValueError(f"DAG `{self.dag_id}` duplicate edge `{u} -> {v}`")
            seen_edges.add((u, v))
            if v not in n_set:
                raise ValueError(f"DAG `{self.dag_id}` edge target `{v}` not in active nodes")
            if u == "__input__":
                has_root_in.add(v)
            else:
                if u not in n_set or u == v:
                    raise ValueError(f"DAG `{self.dag_id}` invalid internal edge `{u} -> {v}`")
                adj[u].append(v)
                rev[v].append(u)
                in_deg[v] += 1

        if not has_root_in:
            raise ValueError(f"DAG `{self.dag_id}` must have at least one edge from `__input__`")

        # Enforce fan-out <= 2 per active node
        fanouts = {n: len(adj[n]) for n in self.nodes}
        max_fanout = max(fanouts.values()) if fanouts else 0
        if max_fanout > M101_MAX_FANOUT:
            raise ValueError(f"DAG `{self.dag_id}` max fan-out {max_fanout} exceeds {M101_MAX_FANOUT}")

        # Kahn topological sort + cycle check + longest-path depth
        q = deque(sorted(n for n in self.nodes if in_deg[n] == 0))
        topo: list[str] = []
        depth_at: dict[str, int] = {n: 1 for n in self.nodes}
        while q:
            cur = q.popleft()
            topo.append(cur)
            for nxt in sorted(adj[cur]):
                depth_at[nxt] = max(depth_at[nxt], depth_at[cur] + 1)
                in_deg[nxt] -= 1
                if in_deg[nxt] == 0:
                    q.append(nxt)
        if len(topo) != len(self.nodes):
            raise ValueError(f"DAG `{self.dag_id}` contains a directed cycle")

        max_depth = max(depth_at.values())
        if max_depth > M101_MAX_DEPTH:
            raise ValueError(f"DAG `{self.dag_id}` depth {max_depth} exceeds {M101_MAX_DEPTH}")

        # Reachability from `__input__` and backward reachability to `sink_node_id` (zero orphan nodes!)
        reach_fwd: set[str] = set()
        dq = deque(sorted(has_root_in))
        while dq:
            cur = dq.popleft()
            if cur in reach_fwd:
                continue
            reach_fwd.add(cur)
            for nxt in adj[cur]:
                dq.append(nxt)
        if reach_fwd != n_set:
            raise ValueError(f"DAG `{self.dag_id}` has nodes unreachable from `__input__`: {sorted(n_set - reach_fwd)}")

        reach_bwd: set[str] = set()
        dq2 = deque([self.sink_node_id])
        while dq2:
            cur = dq2.popleft()
            if cur in reach_bwd:
                continue
            reach_bwd.add(cur)
            for prv in rev[cur]:
                dq2.append(prv)
        if reach_bwd != n_set:
            raise ValueError(f"DAG `{self.dag_id}` has dead nodes not reaching sink `{self.sink_node_id}`: {sorted(n_set - reach_bwd)}")

        # Classify topology signature strictly structurally (for post-hoc analytics ONLY, never fed into prompts!)
        has_diamond_or_branch = max_fanout >= 2
        has_skip = any(
            depth_at[v] - depth_at[u] >= 2
            for u, v in self.edges
            if u != "__input__"
        )
        return {
            "active_node_count": len(self.nodes),
            "depth": max_depth,
            "max_fanout": max_fanout,
            "topo_order": topo,
            "fanout_by_node": fanouts,
            "shared_fanout_nodes": sorted(n for n, fo in fanouts.items() if fo >= 2),
            "has_branching_dag": has_diamond_or_branch,
            "has_skip_edge": has_skip,
        }

    def topology_fingerprint(self) -> str:
        info = self.validate_topology()
        return _canonical_json_sha256(
            {
                "nodes": sorted(self.nodes),
                "edges": sorted([list(e) for e in self.edges]),
                "sink_node_id": self.sink_node_id,
                "node_versions": dict(sorted(self.node_versions.items())),
                "topo_order": info["topo_order"],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dag_id": self.dag_id,
            "nodes": list(self.nodes),
            "edges": [list(e) for e in self.edges],
            "sink_node_id": self.sink_node_id,
            "node_versions": dict(self.node_versions),
            "applicability_guard": deepcopy(self.applicability_guard),
            "priority": int(self.priority),
            "origin_action": self.origin_action,
            "origin_step": int(self.origin_step),
            "is_cross_scoped": bool(self.is_cross_scoped),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExecutableComputationDAG:
        return cls(
            dag_id=str(d["dag_id"]),
            nodes=[str(x) for x in d["nodes"]],
            edges=[(str(e[0]), str(e[1])) for e in d["edges"]],
            sink_node_id=str(d["sink_node_id"]),
            node_versions={str(k): str(v) for k, v in (d.get("node_versions") or {}).items()},
            applicability_guard=deepcopy(d.get("applicability_guard") or {}),
            priority=int(d.get("priority", 10)),
            origin_action=str(d.get("origin_action") or "keep_monolithic"),
            origin_step=int(d.get("origin_step", 0)),
            is_cross_scoped=bool(d.get("is_cross_scoped", False)),
        )


@dataclass
class CutWithdrawalMergeReceipt:
    """Evidence-bound cut withdrawal & `merge_nodes`/`rewire_edge` receipt backed by 3-way external `ProbeReceiptStore` FK (`T01`)."""

    receipt_id: str
    world_id: str
    old_dag_id: str
    new_dag_id: str
    action: str  # 'merge_nodes' | 'rewire_edge'
    prior_positive_probe_id: str
    prior_positive_receipt_sha256: str
    preserved_node_probe_id: str
    preserved_node_receipt_sha256: str
    failing_joint_probe_id: str
    failing_joint_receipt_sha256: str
    new_passing_probe_id: str
    new_passing_receipt_sha256: str
    preserved_old_dag_id: str
    expanded_guard_features: list[str]
    verifier_version: str = SHARED_KERNEL_VERSION

    def compute_expected_receipt_id(self) -> str:
        payload = {k: v for k, v in self.to_dict(include_id=False).items()}
        return "m101wd_" + _canonical_json_sha256(payload)[:24]

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "world_id": self.world_id,
            "old_dag_id": self.old_dag_id,
            "new_dag_id": self.new_dag_id,
            "action": self.action,
            "prior_positive_probe_id": self.prior_positive_probe_id,
            "prior_positive_receipt_sha256": self.prior_positive_receipt_sha256,
            "preserved_node_probe_id": self.preserved_node_probe_id,
            "preserved_node_receipt_sha256": self.preserved_node_receipt_sha256,
            "failing_joint_probe_id": self.failing_joint_probe_id,
            "failing_joint_receipt_sha256": self.failing_joint_receipt_sha256,
            "new_passing_probe_id": self.new_passing_probe_id,
            "new_passing_receipt_sha256": self.new_passing_receipt_sha256,
            "preserved_old_dag_id": self.preserved_old_dag_id,
            "expanded_guard_features": sorted(self.expanded_guard_features),
            "verifier_version": self.verifier_version,
        }
        if include_id:
            d["receipt_id"] = self.receipt_id
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CutWithdrawalMergeReceipt:
        return cls(
            receipt_id=str(d["receipt_id"]),
            world_id=str(d["world_id"]),
            old_dag_id=str(d["old_dag_id"]),
            new_dag_id=str(d["new_dag_id"]),
            action=str(d["action"]),
            prior_positive_probe_id=str(d["prior_positive_probe_id"]),
            prior_positive_receipt_sha256=str(d["prior_positive_receipt_sha256"]),
            preserved_node_probe_id=str(d["preserved_node_probe_id"]),
            preserved_node_receipt_sha256=str(d["preserved_node_receipt_sha256"]),
            failing_joint_probe_id=str(d["failing_joint_probe_id"]),
            failing_joint_receipt_sha256=str(d["failing_joint_receipt_sha256"]),
            new_passing_probe_id=str(d["new_passing_probe_id"]),
            new_passing_receipt_sha256=str(d["new_passing_receipt_sha256"]),
            preserved_old_dag_id=str(d["preserved_old_dag_id"]),
            expanded_guard_features=[str(x) for x in d.get("expanded_guard_features") or []],
            verifier_version=str(d.get("verifier_version") or SHARED_KERNEL_VERSION),
        )


def verify_m101_cut_withdrawal_external_fk(
    receipt: CutWithdrawalMergeReceipt,
    probe_store: ProbeReceiptStore,
) -> dict[str, Any]:
    """Verify ALL external `ProbeReceiptStore` links on a `CutWithdrawalMergeReceipt` (`T01` strict 3+1 FK!)."""
    if receipt.action not in {"merge_nodes", "rewire_edge"}:
        raise ValueError(f"Invalid cut withdrawal action `{receipt.action}`")
    if receipt.receipt_id != receipt.compute_expected_receipt_id():
        raise ValueError(f"Tampered CutWithdrawalMergeReceipt id `{receipt.receipt_id}`")

    e_prior = probe_store.load_probe_receipt(
        receipt.prior_positive_probe_id,
        expected_content_sha256=receipt.prior_positive_receipt_sha256,
        verify_graph_index=True,
    )
    e_pres = probe_store.load_probe_receipt(
        receipt.preserved_node_probe_id,
        expected_content_sha256=receipt.preserved_node_receipt_sha256,
        verify_graph_index=True,
    )
    e_fail = probe_store.load_probe_receipt(
        receipt.failing_joint_probe_id,
        expected_content_sha256=receipt.failing_joint_receipt_sha256,
        verify_graph_index=True,
    )
    e_new = probe_store.load_probe_receipt(
        receipt.new_passing_probe_id,
        expected_content_sha256=receipt.new_passing_receipt_sha256,
        verify_graph_index=True,
    )

    if e_prior["receipt"].get("accepted") is not True:
        raise ValueError("CutWithdrawal prior_positive probe did not pass (`T01`)")
    if e_pres["receipt"].get("healthy") is not True and e_pres["receipt"].get("accepted") is not True:
        raise ValueError("CutWithdrawal preserved_node probe not verified healthy (`T01`)")
    if e_fail["receipt"].get("accepted") is not False:
        raise ValueError("CutWithdrawal failing_joint probe did not fail (`T01`)")
    if e_new["receipt"].get("accepted") is not True:
        raise ValueError("CutWithdrawal new_passing probe did not pass (`T01`)")
    if str(e_prior.get("executed_utc") or "") > str(e_fail.get("executed_utc") or ""):
        raise ValueError("CutWithdrawal temporal ordering violation (`prior_utc > failing_utc`, `T01`)")
    return {
        "verified": True,
        "receipt_id": receipt.receipt_id,
        "verified_probe_ids": [
            receipt.prior_positive_probe_id,
            receipt.preserved_node_probe_id,
            receipt.failing_joint_probe_id,
            receipt.new_passing_probe_id,
        ],
    }


def compile_m101_dag_to_instrumented_solver_code(
    dag: ExecutableComputationDAG,
    node_versions_by_id: dict[str, ExecutableModuleNode],
) -> tuple[str, dict[str, Any]]:
    """Compile an `ExecutableComputationDAG` into a standalone self-contained `batch_solver.py` with runtime
    instrumentation verifying:
      - topological execution order,
      - `exec_count[nid] == 1` for EVERY active node (especially shared `fan-out >= 2` diamond/skip prefixes!),
      - deep-copy anti-aliasing isolation between nodes,
      - anti-identity-shell check (`non-sink` nodes must transform their inputs rather than returning an identical copy!).
    """
    topo_info = dag.validate_topology()
    topo_order: list[str] = topo_info["topo_order"]
    incoming: dict[str, list[str]] = {n: [] for n in dag.nodes}
    for u, v in dag.edges:
        incoming[v].append(u)
    for n in incoming:
        incoming[n].sort()

    node_code_map: dict[str, str] = {}
    node_sha_map: dict[str, str] = {}
    for nid in topo_order:
        vid = dag.node_versions[nid]
        if vid not in node_versions_by_id:
            raise KeyError(f"Unknown node version `{vid}` for node `{nid}` in DAG `{dag.dag_id}`")
        m_node = node_versions_by_id[vid]
        node_code_map[nid] = m_node.code
        node_sha_map[nid] = m_node.code_sha256

    spec_blob = {
        "dag_id": dag.dag_id,
        "topo_order": topo_order,
        "incoming": incoming,
        "sink_node_id": dag.sink_node_id,
        "depth": topo_info["depth"],
        "max_fanout": topo_info["max_fanout"],
        "fanout_by_node": topo_info["fanout_by_node"],
        "shared_fanout_nodes": topo_info["shared_fanout_nodes"],
        "node_code_shas": node_sha_map,
    }
    code_lines = [
        '"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""',
        "from __future__ import annotations",
        "from copy import deepcopy",
        "import hashlib",
        "import json",
        "import math",
        "import os",
        "import numpy as np",
        "",
        f"_DAG_SPEC = {json.dumps(spec_blob, sort_keys=True)!r}",
        f"_NODE_SOURCES = {json.dumps(node_code_map, sort_keys=True)!r}",
        "",
        "def _canon_digest(obj) -> str:",
        "    h = hashlib.sha256()",
        "    def _feed(v):",
        "        if v is None: h.update(b'N;')",
        "        elif isinstance(v, bool): h.update(b'B1;' if v else b'B0;')",
        "        elif isinstance(v, (int, np.integer)): h.update(f'I:{int(v)};'.encode())",
        "        elif isinstance(v, (float, np.floating)):",
        "            fv = float(v)",
        "            if math.isnan(fv): h.update(b'F:NaN;')",
        "            elif math.isinf(fv): h.update(b'F:Inf;')",
        "            else: h.update(b'F:' + np.asarray([0.0 if fv == 0.0 else fv], dtype='<f8').tobytes() + b';')",
        "        elif isinstance(v, str): h.update(f'S:{len(v)}:{v};'.encode())",
        "        elif isinstance(v, np.ndarray):",
        "            h.update(f'A:{v.dtype.kind}:{v.shape}:'.encode())",
        "            if v.dtype.kind == 'f':",
        "                a = np.asarray(v, dtype=np.float64)",
        "                nm = np.isnan(a)",
        "                cl = np.where(nm, 0.0, a)",
        "                cl = np.where(cl == 0.0, 0.0, cl).astype('<f8')",
        "                h.update(cl.tobytes('C'))",
        "            else:",
        "                h.update(np.asarray(v, dtype='<i8').tobytes('C'))",
        "            h.update(b';')",
        "        elif isinstance(v, dict):",
        "            h.update(f'D:{len(v)}:'.encode())",
        "            for k in sorted(v.keys(), key=str): _feed(str(k)); _feed(v[k])",
        "            h.update(b';')",
        "        elif isinstance(v, (list, tuple)):",
        "            h.update(f'L:{len(v)}:'.encode())",
        "            for it in v: _feed(it)",
        "            h.update(b';')",
        "        else:",
        "            h.update(repr(v).encode())",
        "    _feed(obj)",
        "    return h.hexdigest()",
        "",
        "def _load_node_fn(nid: str, src: str):",
        "    ns = {'__name__': f'_hegel_node_{nid}'}",
        "    exec(compile(src, f'<dag_node_{nid}>', 'exec'), ns)",
        "    fn = ns.get('run_node') or ns.get('solve_batch')",
        "    if not callable(fn):",
        "        raise RuntimeError(f'DAG node {nid} missing callable run_node/solve_batch')",
        "    return fn",
        "",
        "_COMPILED_FNS = {nid: _load_node_fn(nid, src) for nid, src in json.loads(_NODE_SOURCES).items()}",
        "_SPEC_OBJ = json.loads(_DAG_SPEC)",
        "",
        "def solve_batch(study: dict) -> dict:",
        "    outputs = {'__input__': deepcopy(study)}",
        "    exec_counts = {nid: 0 for nid in _SPEC_OBJ['topo_order']}",
        "    non_identity_checks = {}",
        "    for nid in _SPEC_OBJ['topo_order']:",
        "        srcs = _SPEC_OBJ['incoming'][nid]",
        "        if len(_SPEC_OBJ['topo_order']) == 1 and srcs == ['__input__']:",
        "            node_in = deepcopy(outputs['__input__'])",
        "        else:",
        "            node_in = {s: deepcopy(outputs[s]) for s in srcs}",
        "        in_dig = _canon_digest(node_in)",
        "        out_v = _COMPILED_FNS[nid](node_in)",
        "        exec_counts[nid] += 1",
        "        if exec_counts[nid] != 1:",
        "            raise RuntimeError(f'DAG node {nid} executed {exec_counts[nid]} times != 1')",
        "        if not isinstance(out_v, dict) or not out_v:",
        "            raise RuntimeError(f'DAG node {nid} must return a non-empty dict')",
        "        out_dig = _canon_digest(out_v)",
        "        if nid != _SPEC_OBJ['sink_node_id']:",
        "            # Anti-identity-shell check: intermediate node cannot merely echo its raw predecessor dict",
        "            echoed = (out_dig == in_dig) or any(out_dig == _canon_digest(outputs[s]) for s in srcs)",
        "            if echoed:",
        "                raise RuntimeError(f'DAG node {nid} failed anti-identity-shell check')",
        "            non_identity_checks[nid] = True",
        "        outputs[nid] = out_v",
        "    trace_fp = os.environ.get('HEGEL_M101_DAG_TRACE_PATH')",
        "    if trace_fp:",
        "        trace_obj = {",
        "            'dag_id': _SPEC_OBJ['dag_id'],",
        "            'active_node_count': len(_SPEC_OBJ['topo_order']),",
        "            'depth': _SPEC_OBJ['depth'],",
        "            'max_fanout': _SPEC_OBJ['max_fanout'],",
        "            'shared_fanout_nodes': _SPEC_OBJ['shared_fanout_nodes'],",
        "            'topo_order': _SPEC_OBJ['topo_order'],",
        "            'exec_counts': exec_counts,",
        "            'all_exec_count_eq_1': all(v == 1 for v in exec_counts.values()),",
        "            'non_identity_checks': non_identity_checks,",
        "            'node_code_shas': _SPEC_OBJ['node_code_shas'],",
        "        }",
        "        open(trace_fp, 'w', encoding='utf-8').write(json.dumps(trace_obj, sort_keys=True))",
        "    return outputs[_SPEC_OBJ['sink_node_id']]",
        "",
    ]
    code_txt = "\n".join(code_lines)
    return code_txt, {**topo_info, "compiled_sha256": hashlib.sha256(code_txt.encode("utf-8")).hexdigest()}


# ============================================================================
# 2. Blind 4-Topology (+ 4 Stratum-B Unseen Subclasses) Scientific World
# ============================================================================


@dataclass(frozen=True)
class M101WorldSpec:
    world_id: str
    cluster_id: str
    split: str  # 'dev' | 'confirmation'
    stratum: str  # 'stratum_a_seen_topology' | 'stratum_b_unseen_topology'
    topology_class: str  # hidden from agent!
    seed_base: int
    telemetry_channels: tuple[int, int, int]  # 3 active channels among 0..3


# `n = 4` Dev Clusters (`1011000..1014000`, `8` continual steps after `Stage S`)
DEV_M101_WORLD_SPECS: tuple[M101WorldSpec, ...] = (
    M101WorldSpec(
        world_id="dev_m101_c01",
        cluster_id="dev_cluster_m101_01",
        split="dev",
        stratum="stratum_a_seen_topology",
        topology_class="monolithic_coupled",
        seed_base=1011000,
        telemetry_channels=(0, 1, 2),
    ),
    M101WorldSpec(
        world_id="dev_m101_c02",
        cluster_id="dev_cluster_m101_02",
        split="dev",
        stratum="stratum_a_seen_topology",
        topology_class="chain_k3_variable_cut",
        seed_base=1012000,
        telemetry_channels=(1, 2, 3),
    ),
    M101WorldSpec(
        world_id="dev_m101_c03",
        cluster_id="dev_cluster_m101_03",
        split="dev",
        stratum="stratum_a_seen_topology",
        topology_class="diamond_k4_shared_prefix",
        seed_base=1013000,
        telemetry_channels=(2, 0, 3),
    ),
    M101WorldSpec(
        world_id="dev_m101_c04",
        cluster_id="dev_cluster_m101_04",
        split="dev",
        stratum="stratum_a_seen_topology",
        topology_class="chain_k3_reversed_cut",
        seed_base=1014000,
        telemetry_channels=(3, 1, 0),
    ),
)

# `n = 8` Unexposed Independent Confirmation Clusters (`1021000..1028000`):
#   - Stratum A (`n = 4`): New independent source parameters in seen topology families (`monolithic`, `chain_k3` x2, `diamond_k4`).
#   - Stratum B (`n = 4`): HELD-OUT UNSEEN topology subclasses / shifted cut depths NEVER present in `dev`!
CONFIRMATION_M101_WORLD_SPECS: tuple[M101WorldSpec, ...] = (
    M101WorldSpec(
        world_id="conf_m101_a01",
        cluster_id="conf_cluster_m101_a01",
        split="confirmation",
        stratum="stratum_a_seen_topology",
        topology_class="monolithic_coupled",
        seed_base=1021000,
        telemetry_channels=(1, 0, 3),
    ),
    M101WorldSpec(
        world_id="conf_m101_a02",
        cluster_id="conf_cluster_m101_a02",
        split="confirmation",
        stratum="stratum_a_seen_topology",
        topology_class="chain_k3_variable_cut",
        seed_base=1022000,
        telemetry_channels=(0, 2, 1),
    ),
    M101WorldSpec(
        world_id="conf_m101_a03",
        cluster_id="conf_cluster_m101_a03",
        split="confirmation",
        stratum="stratum_a_seen_topology",
        topology_class="chain_k3_reversed_cut",
        seed_base=1023000,
        telemetry_channels=(2, 3, 0),
    ),
    M101WorldSpec(
        world_id="conf_m101_a04",
        cluster_id="conf_cluster_m101_a04",
        split="confirmation",
        stratum="stratum_a_seen_topology",
        topology_class="diamond_k4_shared_prefix",
        seed_base=1024000,
        telemetry_channels=(3, 2, 1),
    ),
    M101WorldSpec(
        world_id="conf_m101_b01",
        cluster_id="conf_cluster_m101_b01",
        split="confirmation",
        stratum="stratum_b_unseen_topology",
        topology_class="chain_k4_deep_cascade",
        seed_base=1025000,
        telemetry_channels=(0, 3, 2),
    ),
    M101WorldSpec(
        world_id="conf_m101_b02",
        cluster_id="conf_cluster_m101_b02",
        split="confirmation",
        stratum="stratum_b_unseen_topology",
        topology_class="asymmetric_skip_dag_k4",
        seed_base=1026000,
        telemetry_channels=(1, 3, 0),
    ),
    M101WorldSpec(
        world_id="conf_m101_b03",
        cluster_id="conf_cluster_m101_b03",
        split="confirmation",
        stratum="stratum_b_unseen_topology",
        topology_class="diamond_k4_coupled_first",
        seed_base=1027000,
        telemetry_channels=(2, 1, 3),
    ),
    M101WorldSpec(
        world_id="conf_m101_b04",
        cluster_id="conf_cluster_m101_b04",
        split="confirmation",
        stratum="stratum_b_unseen_topology",
        topology_class="hybrid_mono_to_split_shift",
        seed_base=1028000,
        telemetry_channels=(3, 0, 1),
    ),
)


_M101_WORLD_GEN_PY = r'''
import math
import numpy as np


def _m101_apply_pa(z: np.ndarray, mode: str, gains: np.ndarray, offsets: np.ndarray, mix: float) -> np.ndarray:
    """Strictly non-nested forward sensor map `z -> x_raw` (`panel_a` has `(u=z_ref, v=x_raw)` with 3 outliers)."""
    z2 = np.asarray(z, dtype=float)
    y = z2 * gains[None, :] + offsets[None, :]
    if mode == "A0":
        return y
    if mode == "A1":
        # Adjacent-channel bilinear bridge + cubic coupling (linear inverse fails MAE > 0.25!)
        z_roll = np.roll(z2, -1, axis=1)
        return y + mix * z_roll + 0.32 * (z2 * z_roll) + 0.16 * (z2 ** 3)
    # A2: opposite-channel high-frequency sinusoidal saturation
    z_opp = np.roll(z2, 2, axis=1)
    return y + mix * np.sin(2.4 * z_opp) + 0.18 * np.tanh(1.8 * z2)


def _m101_apply_pb(z: np.ndarray, mode: str, W_mat: np.ndarray, b_vec: np.ndarray) -> np.ndarray:
    """Strictly non-nested operator stage `z -> h` (`panel_b` has 24 probe rows -> generic 4D poly overfits!)."""
    z2 = np.asarray(z, dtype=float)
    if mode == "B0":
        return z2 @ W_mat.T + b_vec[None, :]
    if mode == "B1":
        # Non-smooth dead-zone power-1.5 activation (tau=0.35): strictly non-nested with linear & cosine!
        dz = np.sign(z2) * (np.maximum(np.abs(z2) - 0.35, 0.0) ** 1.5)
        return 1.35 * (dz @ W_mat.T) + b_vec[None, :]
    # B2: high-frequency dispersive cosine operator
    return (np.cos(2.6 * z2)) @ W_mat.T + b_vec[None, :]


def _m101_apply_pc_branch(z: np.ndarray, mode: str, W_mat: np.ndarray, b_vec: np.ndarray) -> np.ndarray:
    """Strictly non-nested parallel branch / 3rd-stage operators (`panel_c` has 24 probe rows!)."""
    z2 = np.asarray(z, dtype=float)
    if mode in ("L0", "H0"):
        # Monotone exponential saturation branch
        act = np.sign(z2) * (1.0 - np.exp(-1.8 * np.abs(z2)))
        return 1.25 * (act @ W_mat.T) + b_vec[None, :]
    if mode in ("L1", "H1"):
        # Cross-channel adjacent bilinear product (strictly non-nested with ANY 1D elementwise curve fitter!)
        act = z2 * np.roll(z2, -1, axis=1)
        return 1.45 * (act @ W_mat.T) + b_vec[None, :]
    if mode == "R0":
        # Asymmetric softplus energy branch
        act = np.log1p(np.exp(2.2 * z2)) / 2.2 - 0.315
        return 1.15 * (act @ W_mat.T) + b_vec[None, :]
    # R1: strictly even high-frequency cosine operator (zero linear/odd component -> non-nested with monotone R0!)
    act = np.cos(2.4 * z2)
    return 1.25 * (act @ W_mat.T) + b_vec[None, :]


def _m101_readout(
    h_main: np.ndarray,
    h_aux: np.ndarray | None,
    z_cal: np.ndarray,
    eval_t: np.ndarray,
    mode: str,
    c_y: np.ndarray,
    c_tr: np.ndarray,
    cross_coupled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Non-nested Readout / Conservation Join returning clean `(y, traj, summary)`."""
    if h_aux is None:
        base_y = h_main @ c_y + 0.18 * np.sum(h_main[:, :2] * h_main[:, 2:], axis=1)
        amp_a = h_main @ c_tr[:, 0]
        amp_b = h_main @ c_tr[:, 1]
    else:
        # True bilinear conservation join combining BOTH branches (`h_main`, `h_aux`)!
        bilin = np.sum(h_main * h_aux, axis=1)
        base_y = 0.72 * (h_main @ c_y) + 0.68 * (h_aux @ c_y[::-1]) + 0.42 * bilin
        amp_a = h_main @ c_tr[:, 0] + 0.35 * bilin
        amp_b = h_aux @ c_tr[:, 1] - 0.25 * (h_main[:, 0] * h_aux[:, 1])

    if mode == "C0":
        phi_a = 1.0 - np.exp(-eval_t / 0.55)
        phi_b = eval_t / (1.0 + 0.40 * eval_t)
        y = base_y
    elif mode == "C1":
        phi_a = np.log1p(1.4 * eval_t)
        phi_b = 1.0 - np.exp(-eval_t / 1.45)
        y = base_y + 0.45 * np.tanh(1.6 * base_y)
    else:
        phi_a = np.sin(1.4 * eval_t) * np.exp(-0.25 * eval_t)
        phi_b = np.sqrt(1.0 + eval_t) - 1.0
        y = 0.85 * base_y + 0.35 * (base_y ** 2) / (1.0 + 0.3 * (base_y ** 2))

    traj = amp_a[:, None] * phi_a[None, :] + amp_b[:, None] * phi_b[None, :]
    if cross_coupled:
        # Genuine cross-stage physical interaction breaking separability of earlier cut!
        cpl = 0.70 * np.sum(z_cal * np.roll(h_main, 1, axis=1), axis=1)
        y = y + cpl
        traj = traj + cpl[:, None] * (0.55 * np.sqrt(eval_t)[None, :])

    summary = np.array([float(np.mean(y)), float(np.std(y)), float(np.mean(traj[:, -1])), float(np.std(traj[:, -1]))], dtype=float)
    return y, traj, summary


def _m101_monolithic_core(
    x: np.ndarray,
    eval_t: np.ndarray,
    core_mode: str,
    W1: np.ndarray,
    W2: np.ndarray,
    c_y: np.ndarray,
    c_tr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inseparable implicit fixed-point system `u = tanh(x W1^T + alpha u W2^T)` where NO auxiliary panel factorizes `x -> out`!"""
    n = x.shape[0]
    u = np.zeros((n, 4), dtype=float)
    alpha = 0.38 if core_mode == "M0" else (-0.36 if core_mode == "M1" else 0.44)
    for _ in range(10):
        u_next = np.tanh(0.78 * (x @ W1.T) + alpha * (u @ W2.T) + (0.20 * (u * x) if core_mode == "M2" else 0.0))
        u = 0.5 * u + 0.5 * u_next
    quad = np.sum(u * x, axis=1)
    y = u @ c_y + 0.35 * quad + (0.28 * np.sin(1.4 * quad) if core_mode != "M0" else 0.0)
    a1 = u @ c_tr[:, 0] + 0.22 * quad
    a2 = (x @ c_tr[:, 1]) * (0.5 + 0.5 * np.tanh(u[:, 0]))
    if core_mode == "M0":
        p1 = 1.0 - np.exp(-eval_t / 0.65)
        p2 = np.log1p(eval_t)
    elif core_mode == "M1":
        p1 = eval_t / (1.0 + 0.7 * eval_t)
        p2 = 1.0 - np.exp(-eval_t / 1.3)
    else:
        p1 = np.sqrt(1.0 + eval_t) - 1.0
        p2 = np.sin(1.1 * eval_t) * np.exp(-0.2 * eval_t)
    traj = a1[:, None] * p1[None, :] + a2[:, None] * p2[None, :]
    summary = np.array([float(np.mean(y)), float(np.std(y)), float(np.mean(traj[:, -1])), float(np.std(traj[:, -1]))], dtype=float)
    return y, traj, summary


def _gen_m101_study(
    seed: int,
    *,
    world_seed: int,
    topology_kind: str,
    stage_cfg: dict,
    telemetry_vec: list,
):
    """Generate one blind scientific study with IDENTICAL schema across ALL topologies."""
    w_rng = np.random.default_rng(world_seed)
    gains = w_rng.uniform(0.82, 1.24, size=4)
    offsets = w_rng.uniform(-0.18, 0.18, size=4)
    mix_pa = float(w_rng.choice([-0.32, 0.32]))

    def _orth_mat(scale: float) -> np.ndarray:
        q, _ = np.linalg.qr(w_rng.normal(size=(4, 4)))
        return scale * q

    W_b = _orth_mat(0.88)
    b_b = w_rng.uniform(-0.15, 0.15, size=4)
    W_l = _orth_mat(0.84)
    b_l = w_rng.uniform(-0.15, 0.15, size=4)
    W_r = _orth_mat(0.84)
    b_r = w_rng.uniform(-0.15, 0.15, size=4)
    W_m1 = _orth_mat(0.78)
    W_m2 = _orth_mat(0.72)
    c_y = w_rng.uniform(0.50, 0.95, size=4) * w_rng.choice([-1.0, 1.0], size=4)
    c_tr = w_rng.uniform(0.40, 0.85, size=(4, 2)) * w_rng.choice([-1.0, 1.0], size=(4, 2))

    rng = np.random.default_rng(seed)
    eval_t = np.linspace(0.25, 3.0, 8, dtype=float)
    n_tr, n_te = 72, 36

    pa_mode = str(stage_cfg.get("pa", "A0"))
    pb_mode = str(stage_cfg.get("pb", "B0"))
    pc1_mode = str(stage_cfg.get("pc1", "L0"))
    pc2_mode = str(stage_cfg.get("pc2", "R0"))
    rd_mode = str(stage_cfg.get("rd", "C0"))
    core_mode = str(stage_cfg.get("core", "M0"))
    cross_cpl = bool(stage_cfg.get("cross_coupled", False))

    # Public panels (`panel_a`: 48 rows with 3 outliers; `panel_b`: 24 rows; `panel_c`: 24 rows)
    # 24 rows cleanly identifies a 4x4 operator + 4 bias (20 linear params) ONCE the true nonlinear law is known,
    # while blocking blind 4D degree-3 polynomial overfitting (35+ params > 24 rows)!
    pa_u = rng.uniform(-1.3, 1.3, size=(48, 4))
    pa_v = _m101_apply_pa(pa_u, pa_mode, gains, offsets, mix_pa) + rng.normal(0.0, 1e-4, size=(48, 4))
    pa_v[5] += np.array([6.5, -5.8, 6.2, -6.0])
    pa_v[17] += np.array([-6.1, 6.4, -5.9, 6.3])
    pa_v[31] += np.array([5.7, 6.1, -6.4, -5.8])

    pb_u = rng.uniform(-1.25, 1.25, size=(24, 4))
    pb_v = _m101_apply_pb(pb_u, pb_mode, W_b, b_b) + rng.normal(0.0, 1e-4, size=(24, 4))

    pc_u = rng.uniform(-1.25, 1.25, size=(24, 4))
    pc_v1 = _m101_apply_pc_branch(pc_u, pc1_mode, W_l, b_l) + rng.normal(0.0, 1e-4, size=(24, 4))
    pc_v2 = _m101_apply_pc_branch(pc_u, pc2_mode, W_r, b_r) + rng.normal(0.0, 1e-4, size=(24, 4))

    def _eval_split(n_rows: int):
        z_true = rng.uniform(-1.15, 1.15, size=(n_rows, 4))
        if topology_kind == "monolithic":
            x_raw = z_true
            y_cl, tr_cl, sm_cl = _m101_monolithic_core(x_raw, eval_t, core_mode, W_m1, W_m2, c_y, c_tr)
            return x_raw, y_cl, tr_cl, sm_cl

        x_raw = _m101_apply_pa(z_true, pa_mode, gains, offsets, mix_pa)
        if topology_kind == "chain_k3":
            h1 = _m101_apply_pb(z_true, pb_mode, W_b, b_b)
            y_cl, tr_cl, sm_cl = _m101_readout(h1, None, z_true, eval_t, rd_mode, c_y, c_tr, cross_cpl)
        elif topology_kind == "chain_k4":
            h1 = _m101_apply_pb(z_true, pb_mode, W_b, b_b)
            h2 = _m101_apply_pc_branch(h1, pc1_mode, W_l, b_l)
            y_cl, tr_cl, sm_cl = _m101_readout(h2, None, z_true, eval_t, rd_mode, c_y, c_tr, cross_cpl)
        elif topology_kind == "diamond_k4":
            u_l = _m101_apply_pc_branch(z_true, pc1_mode, W_l, b_l)
            u_r = _m101_apply_pc_branch(z_true, pc2_mode, W_r, b_r)
            y_cl, tr_cl, sm_cl = _m101_readout(u_l, u_r, z_true, eval_t, rd_mode, c_y, c_tr, cross_cpl)
        elif topology_kind == "skip_dag_k4":
            h1 = _m101_apply_pb(z_true, pb_mode, W_b, b_b)
            h2 = _m101_apply_pc_branch(h1, pc1_mode, W_l, b_l)
            y_cl, tr_cl, sm_cl = _m101_readout(h2, z_true, z_true, eval_t, rd_mode, c_y, c_tr, cross_cpl)
        else:
            raise ValueError(f"Unknown topology_kind: {topology_kind}")
        return x_raw, y_cl, tr_cl, sm_cl

    tr_x, tr_y_cl, tr_tr_cl, tr_sm_cl = _eval_split(n_tr)
    te_x, te_y_cl, te_tr_cl, te_sm_cl = _eval_split(n_te)

    tr_y_obs = tr_y_cl + rng.normal(0.0, 0.005, size=tr_y_cl.shape)
    tr_tr_obs = tr_tr_cl + rng.normal(0.0, 0.005, size=tr_tr_cl.shape)

    study = {
        "eval_times": eval_t,
        "panel_a": {"u": pa_u, "v": pa_v},
        "panel_b": {"u": pb_u, "v": pb_v},
        "panel_c": {"u": pc_u, "v1": pc_v1, "v2": pc_v2},
        "train": {"x": tr_x, "y": tr_y_obs, "traj": tr_tr_obs, "summary": tr_sm_cl},
        "test": {"x": te_x},
        "telemetry_vec": np.asarray(telemetry_vec, dtype=float),
    }
    gt = {
        "y_clean": te_y_cl,
        "traj_clean": te_tr_cl,
        "summary_clean": te_sm_cl,
    }
    return study, gt
'''


BLIND_M101_INSTRUCTION_TEXT = (
    "You are solving a blind numerical scientific prediction task in `/holdout_task`.\n"
    "Load `/holdout_task/public_batch.pkl` (a 1-element list whose element is `{'study': study_dict}`).\n"
    "`study_dict` contains:\n"
    "  - `eval_times`: shape `(8,)` evaluation timestamps.\n"
    "  - `panel_a`: `{'u': (48, 4), 'v': (48, 4)}` calibration/reference panel (may contain sparse outliers).\n"
    "  - `panel_b`: `{'u': (24, 4), 'v': (24, 4)}` auxiliary operator panel.\n"
    "  - `panel_c`: `{'u': (24, 4), 'v1': (24, 4), 'v2': (24, 4)}` dual-output auxiliary operator panel.\n"
    "  - `train`: `{'x': (72, 4), 'y': (72,), 'traj': (72, 8), 'summary': (4,)}`.\n"
    "  - `test`: `{'x': (36, 4)}`.\n"
    "  - `telemetry_vec`: shape `(4,)` apparatus state vector.\n"
    "Note: Not every panel is necessarily active in the forward physical law of a given campaign; infer the true "
    "mathematical mapping strictly from the data in `public_batch.pkl` and validate via internal held-out cross-validation "
    "(e.g., fitting on `train[:56]` and checking MAE on `train[56:]`).\n"
    "Write `/holdout_task/batch_solver.py` exposing `solve_batch(study: dict) -> dict` returning:\n"
    "  - `'y_hat'`: finite `np.ndarray` of shape `(len(study['test']['x']),)`\n"
    "  - `'state_pred'`: finite `np.ndarray` of shape `(len(study['test']['x']), len(study['eval_times']))`\n"
    "  - `'diagnostics_vec'`: finite `np.ndarray` of shape `(4,)` equal to "
    "`[mean(y_hat), std(y_hat), mean(state_pred[:, -1]), std(state_pred[:, -1])]`.\n"
    "Run `python3 /holdout_task/validate_public_schema.py` before finishing."
)


def compute_m101_neutral_observables(study: dict[str, Any]) -> dict[str, float | None]:
    """Extract uniform neutral 18-dim observables `obs_00..obs_17` purely from public `study` (`0` topology/role hints!)."""
    pa = study["panel_a"]
    pb = study["panel_b"]
    pc = study["panel_c"]
    tr = study["train"]

    def _fit_panel_stats(u: np.ndarray, v: np.ndarray) -> tuple[float, float, float, float]:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        # Robustly filter sparse outliers via initial median-residual mask
        aug = np.column_stack([u, np.ones(len(u))])
        w = np.ones(len(u))
        coef = np.zeros((5, 4))
        for _ in range(5):
            sw = np.sqrt(w)[:, None]
            coef, *_ = np.linalg.lstsq(aug * sw, v * sw, rcond=None)
            res = np.linalg.norm(v - aug @ coef, axis=1)
            med = max(float(np.median(res)), 1e-6)
            w = 1.0 / (1.0 + (res / (3.0 * med)) ** 2)
        inliers = np.linalg.norm(v - aug @ coef, axis=1) <= max(5.0 * float(np.median(np.linalg.norm(v - aug @ coef, axis=1))), 0.15)
        ui, vi = u[inliers], v[inliers]
        aug_i = np.column_stack([ui, np.ones(len(ui))])
        c_lin, *_ = np.linalg.lstsq(aug_i, vi, rcond=None)
        res_lin = vi - aug_i @ c_lin
        ss_tot = np.maximum(np.sum((vi - np.mean(vi, axis=0)) ** 2, axis=0), 1e-12)
        r2_lin = float(np.min(1.0 - np.sum(res_lin ** 2, axis=0) / ss_tot))

        # Off-diagonal linear ratio
        off_r = []
        for j in range(4):
            d_j = max(abs(float(c_lin[j, j])), 1e-4)
            off_r.append(max(abs(float(c_lin[k, j])) for k in range(4) if k != j) / d_j)
        cross_max = float(max(off_r))

        # Cubic/quadratic + harmonic nonlinear gain
        phi_nl = np.column_stack([aug_i, ui ** 2, ui ** 3, np.sin(1.3 * ui)])
        c_nl, *_ = np.linalg.lstsq(phi_nl, vi, rcond=None)
        res_nl = vi - phi_nl @ c_nl
        r2_nl = float(np.min(1.0 - np.sum(res_nl ** 2, axis=0) / ss_tot))
        nl_gain = max(0.0, r2_nl - r2_lin)
        rms = float(np.sqrt(np.mean(vi ** 2)))
        return round(r2_lin, 6), round(cross_max, 6), round(nl_gain, 6), round(rms, 6)

    pa_r2, pa_cross, pa_nl, pa_rms = _fit_panel_stats(pa["u"], pa["v"])
    pb_r2, _, pb_nl, pb_rms = _fit_panel_stats(pb["u"], pb["v"])
    # Harmonic gain specifically on `panel_b`
    pbu, pbv = np.asarray(pb["u"], dtype=float), np.asarray(pb["v"], dtype=float)
    phi_q = np.column_stack([pbu, np.ones(len(pbu)), pbu ** 2])
    phi_h = np.column_stack([phi_q, np.sin(1.3 * pbu), np.cos(0.9 * pbu)])
    cq, *_ = np.linalg.lstsq(phi_q, pbv, rcond=None)
    ch, *_ = np.linalg.lstsq(phi_h, pbv, rcond=None)
    ss_pb = max(float(np.sum((pbv - np.mean(pbv, axis=0)) ** 2)), 1e-12)
    pb_harm = max(0.0, float(np.sum((pbv - phi_q @ cq) ** 2) - np.sum((pbv - phi_h @ ch) ** 2)) / ss_pb)

    pc1_r2, _, pc1_nl, _ = _fit_panel_stats(pc["u"], pc["v1"])
    pc2_r2, _, pc2_nl, _ = _fit_panel_stats(pc["u"], pc["v2"])

    tr_y = np.asarray(tr["y"], dtype=float)
    tr_traj = np.asarray(tr["traj"], dtype=float)
    curv = float(np.mean(np.abs(tr_traj[:, -1] - 2.0 * tr_traj[:, len(tr_traj[0]) // 2] + tr_traj[:, 0])))
    tel = np.asarray(study.get("telemetry_vec", np.zeros(4)), dtype=float).ravel()

    return {
        "obs_00_pa_lin_r2": pa_r2,
        "obs_01_pa_cross_corr": pa_cross,
        "obs_02_pa_cubic_gain": pa_nl,
        "obs_03_pa_rms": pa_rms,
        "obs_04_pb_lin_r2": pb_r2,
        "obs_05_pb_quad_gain": pb_nl,
        "obs_06_pb_harmonic_gain": round(pb_harm, 6),
        "obs_07_pb_rms": pb_rms,
        "obs_08_pc_v1_lin_r2": pc1_r2,
        "obs_09_pc_v1_nl_gain": pc1_nl,
        "obs_10_pc_v2_lin_r2": pc2_r2,
        "obs_11_pc_v2_nl_gain": pc2_nl,
        "obs_12_tr_y_std": round(float(np.std(tr_y)), 6),
        "obs_13_tr_traj_curv": round(curv, 6),
        "obs_14_tel_ch0": round(float(tel[0]), 6),
        "obs_15_tel_ch1": round(float(tel[1]), 6),
        "obs_16_tel_ch2": round(float(tel[2]), 6),
        "obs_17_tel_ch3": round(float(tel[3]), 6),
    }


def _schedule_for_topology(spec: M101WorldSpec) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build Step 0 (`Stage S`) + 8 continual steps for each topology class (covering all 4 GPT-6 §D3 contracts!)."""
    ch0, ch1, ch2 = spec.telemetry_channels

    def _tel(v0: float, v1: float, v2: float) -> list[float]:
        arr = [0.1, 0.1, 0.1, 0.1]
        arr[ch0] = v0
        arr[ch1] = v1
        arr[ch2] = v2
        return arr

    tc = spec.topology_class
    if tc == "monolithic_coupled":
        s0 = {"topology_kind": "monolithic", "stage_cfg": {"core": "M0", "pa": "A0", "pb": "B0", "pc1": "L0", "pc2": "R0"}, "tel": _tel(0.2, 0.2, 0.2), "event": "S00_mono_init"}
        steps = [
            {"topology_kind": "monolithic", "stage_cfg": {"core": "M1", "pa": "A0", "pb": "B0", "pc1": "L0", "pc2": "R0"}, "tel": _tel(1.6, 0.2, 0.2), "event": "S01_mono_shift_1"},
            {"topology_kind": "monolithic", "stage_cfg": {"core": "M2", "pa": "A0", "pb": "B0", "pc1": "L0", "pc2": "R0"}, "tel": _tel(0.2, 1.7, 0.2), "event": "S02_mono_shift_2"},
            {"topology_kind": "monolithic", "stage_cfg": {"core": "M0", "pa": "A0", "pb": "B0", "pc1": "L0", "pc2": "R0"}, "tel": _tel(0.2, 0.2, 0.2), "event": "S03_mono_revisit_0"},
            {"topology_kind": "monolithic", "stage_cfg": {"core": "M1", "pa": "A0", "pb": "B0", "pc1": "L0", "pc2": "R0"}, "tel": _tel(1.6, 0.2, 0.2), "event": "S04_mono_revisit_1"},
            {"topology_kind": "monolithic", "stage_cfg": {"core": "M2", "pa": "A0", "pb": "B0", "pc1": "L0", "pc2": "R0"}, "tel": _tel(0.2, 1.7, 0.2), "event": "S05_mono_revisit_2"},
            {"topology_kind": "monolithic", "stage_cfg": {"core": "M1", "pa": "A0", "pb": "B0", "pc1": "L0", "pc2": "R0"}, "tel": _tel(1.6, 0.2, 0.2), "event": "S06_mono_revisit_1b"},
            {"topology_kind": "monolithic", "stage_cfg": {"core": "M0", "pa": "A0", "pb": "B0", "pc1": "L0", "pc2": "R0"}, "tel": _tel(0.2, 0.2, 0.2), "event": "S07_mono_revisit_0b"},
            {"topology_kind": "monolithic", "stage_cfg": {"core": "M2", "pa": "A0", "pb": "B0", "pc1": "L0", "pc2": "R0"}, "tel": _tel(0.2, 1.7, 0.2), "event": "S08_mono_revisit_2b"},
        ]
        return s0, steps

    if tc == "chain_k3_variable_cut":
        s0 = {"topology_kind": "chain_k3", "stage_cfg": {"pa": "A0", "pb": "B0", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S00_k3_base"}
        steps = [
            {"topology_kind": "chain_k3", "stage_cfg": {"pa": "A0", "pb": "B1", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S01_shift_mid"},
            {"topology_kind": "chain_k3", "stage_cfg": {"pa": "A0", "pb": "B0", "rd": "C1", "cross_coupled": False}, "tel": _tel(1.7, 0.2, 0.2), "event": "S02_shift_tail"},
            {"topology_kind": "chain_k3", "stage_cfg": {"pa": "A1", "pb": "B0", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S03_shift_head"},
            {"topology_kind": "chain_k3", "stage_cfg": {"pa": "A1", "pb": "B1", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S04_comb_head_mid"},
            {"topology_kind": "chain_k3", "stage_cfg": {"pa": "A0", "pb": "B1", "rd": "C1", "cross_coupled": False}, "tel": _tel(1.7, 0.2, 0.2), "event": "S05_comb_mid_tail"},
            # Cross-stage interaction `A1 x C1` colliding on identical `tel = (1.7, 0.2, 0.2)` -> forces `merge/rewire` + selective guard expansion!
            {"topology_kind": "chain_k3", "stage_cfg": {"pa": "A1", "pb": "B1", "rd": "C1", "cross_coupled": True}, "tel": _tel(1.7, 0.2, 0.2), "event": "S06_interaction_merge"},
            {"topology_kind": "chain_k3", "stage_cfg": {"pa": "A0", "pb": "B1", "rd": "C1", "cross_coupled": False}, "tel": _tel(1.7, 0.2, 0.2), "event": "S07_revisit_old_sep"},
            {"topology_kind": "chain_k3", "stage_cfg": {"pa": "A1", "pb": "B1", "rd": "C1", "cross_coupled": True}, "tel": _tel(1.7, 0.2, 0.2), "event": "S08_revisit_merged"},
        ]
        return s0, steps

    if tc == "chain_k3_reversed_cut":
        s0 = {"topology_kind": "chain_k3", "stage_cfg": {"pa": "A0", "pb": "B0", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S00_k3r_base"}
        steps = [
            {"topology_kind": "chain_k3", "stage_cfg": {"pa": "A0", "pb": "B0", "rd": "C1", "cross_coupled": False}, "tel": _tel(1.7, 0.2, 0.2), "event": "S01_shift_tail"},
            {"topology_kind": "chain_k3", "stage_cfg": {"pa": "A1", "pb": "B0", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S02_shift_head"},
            {"topology_kind": "chain_k3", "stage_cfg": {"pa": "A0", "pb": "B1", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S03_shift_mid"},
            {"topology_kind": "chain_k3", "stage_cfg": {"pa": "A1", "pb": "B1", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S04_comb_head_mid"},
            {"topology_kind": "chain_k3", "stage_cfg": {"pa": "A0", "pb": "B1", "rd": "C1", "cross_coupled": False}, "tel": _tel(1.7, 0.2, 0.2), "event": "S05_comb_mid_tail"},
            {"topology_kind": "chain_k3", "stage_cfg": {"pa": "A1", "pb": "B0", "rd": "C1", "cross_coupled": True}, "tel": _tel(1.7, 0.2, 0.2), "event": "S06_interaction_merge"},
            {"topology_kind": "chain_k3", "stage_cfg": {"pa": "A0", "pb": "B0", "rd": "C1", "cross_coupled": False}, "tel": _tel(1.7, 0.2, 0.2), "event": "S07_revisit_old_sep"},
            {"topology_kind": "chain_k3", "stage_cfg": {"pa": "A1", "pb": "B0", "rd": "C1", "cross_coupled": True}, "tel": _tel(1.7, 0.2, 0.2), "event": "S08_revisit_merged"},
        ]
        return s0, steps

    if tc == "diamond_k4_shared_prefix":
        s0 = {"topology_kind": "diamond_k4", "stage_cfg": {"pa": "A0", "pc1": "L0", "pc2": "R0", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S00_dia_base"}
        steps = [
            {"topology_kind": "diamond_k4", "stage_cfg": {"pa": "A0", "pc1": "L1", "pc2": "R0", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S01_shift_left"},
            {"topology_kind": "diamond_k4", "stage_cfg": {"pa": "A0", "pc1": "L0", "pc2": "R1", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S02_shift_right"},
            {"topology_kind": "diamond_k4", "stage_cfg": {"pa": "A1", "pc1": "L0", "pc2": "R0", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S03_shift_prefix"},
            {"topology_kind": "diamond_k4", "stage_cfg": {"pa": "A0", "pc1": "L1", "pc2": "R1", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S04_comb_lr"},
            {"topology_kind": "diamond_k4", "stage_cfg": {"pa": "A1", "pc1": "L1", "pc2": "R0", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S05_comb_pl"},
            # Cross-branch interaction `A1 x R1` colliding on identical `tel` -> triggers `merge_nodes` / `rewire_edge`!
            {"topology_kind": "diamond_k4", "stage_cfg": {"pa": "A1", "pc1": "L1", "pc2": "R1", "rd": "C0", "cross_coupled": True}, "tel": _tel(0.2, 0.2, 0.2), "event": "S06_interaction_merge"},
            {"topology_kind": "diamond_k4", "stage_cfg": {"pa": "A0", "pc1": "L1", "pc2": "R1", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S07_revisit_old_dia"},
            {"topology_kind": "diamond_k4", "stage_cfg": {"pa": "A1", "pc1": "L1", "pc2": "R1", "rd": "C0", "cross_coupled": True}, "tel": _tel(0.2, 0.2, 0.2), "event": "S08_revisit_merged"},
        ]
        return s0, steps

    if tc == "chain_k4_deep_cascade":
        # Stratum B HELD-OUT depth-4 linear cascade (`A -> B -> H -> C`)!
        s0 = {"topology_kind": "chain_k4", "stage_cfg": {"pa": "A0", "pb": "B0", "pc1": "H0", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S00_k4_base"}
        steps = [
            {"topology_kind": "chain_k4", "stage_cfg": {"pa": "A0", "pb": "B1", "pc1": "H0", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S01_shift_s2"},
            {"topology_kind": "chain_k4", "stage_cfg": {"pa": "A0", "pb": "B0", "pc1": "H1", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S02_shift_s3"},
            {"topology_kind": "chain_k4", "stage_cfg": {"pa": "A1", "pb": "B0", "pc1": "H0", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S03_shift_s1"},
            {"topology_kind": "chain_k4", "stage_cfg": {"pa": "A0", "pb": "B0", "pc1": "H0", "rd": "C1", "cross_coupled": False}, "tel": _tel(1.7, 0.2, 0.2), "event": "S04_shift_s4"},
            {"topology_kind": "chain_k4", "stage_cfg": {"pa": "A1", "pb": "B1", "pc1": "H1", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S05_comb_123"},
            {"topology_kind": "chain_k4", "stage_cfg": {"pa": "A1", "pb": "B1", "pc1": "H1", "rd": "C1", "cross_coupled": True}, "tel": _tel(1.7, 0.2, 0.2), "event": "S06_interaction_merge"},
            {"topology_kind": "chain_k4", "stage_cfg": {"pa": "A0", "pb": "B1", "pc1": "H1", "rd": "C1", "cross_coupled": False}, "tel": _tel(1.7, 0.2, 0.2), "event": "S07_revisit_old_sep"},
            {"topology_kind": "chain_k4", "stage_cfg": {"pa": "A1", "pb": "B1", "pc1": "H1", "rd": "C1", "cross_coupled": True}, "tel": _tel(1.7, 0.2, 0.2), "event": "S08_revisit_merged"},
        ]
        return s0, steps

    if tc == "asymmetric_skip_dag_k4":
        # Stratum B HELD-OUT skip-connection DAG (`P -> M1 -> M2 -> J` + direct skip `P -> J`, `fan-out=2`)!
        s0 = {"topology_kind": "skip_dag_k4", "stage_cfg": {"pa": "A0", "pb": "B0", "pc1": "H0", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S00_skip_base"}
        steps = [
            {"topology_kind": "skip_dag_k4", "stage_cfg": {"pa": "A0", "pb": "B1", "pc1": "H0", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S01_shift_m1"},
            {"topology_kind": "skip_dag_k4", "stage_cfg": {"pa": "A0", "pb": "B0", "pc1": "H1", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S02_shift_m2"},
            {"topology_kind": "skip_dag_k4", "stage_cfg": {"pa": "A1", "pb": "B0", "pc1": "H0", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S03_shift_prefix"},
            {"topology_kind": "skip_dag_k4", "stage_cfg": {"pa": "A0", "pb": "B1", "pc1": "H1", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S04_comb_m1m2"},
            {"topology_kind": "skip_dag_k4", "stage_cfg": {"pa": "A1", "pb": "B1", "pc1": "H0", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S05_comb_pm1"},
            {"topology_kind": "skip_dag_k4", "stage_cfg": {"pa": "A1", "pb": "B1", "pc1": "H1", "rd": "C0", "cross_coupled": True}, "tel": _tel(0.2, 0.2, 0.2), "event": "S06_interaction_merge"},
            {"topology_kind": "skip_dag_k4", "stage_cfg": {"pa": "A0", "pb": "B1", "pc1": "H1", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S07_revisit_old_skip"},
            {"topology_kind": "skip_dag_k4", "stage_cfg": {"pa": "A1", "pb": "B1", "pc1": "H1", "rd": "C0", "cross_coupled": True}, "tel": _tel(0.2, 0.2, 0.2), "event": "S08_revisit_merged"},
        ]
        return s0, steps

    if tc == "diamond_k4_coupled_first":
        # Stratum B HELD-OUT coupled-first diamond (encounters coupled interaction FIRST, then separable breaks it!)
        s0 = {"topology_kind": "diamond_k4", "stage_cfg": {"pa": "A0", "pc1": "L0", "pc2": "R0", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S00_diacf_base"}
        steps = [
            {"topology_kind": "diamond_k4", "stage_cfg": {"pa": "A1", "pc1": "L0", "pc2": "R0", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S01_shift_prefix"},
            {"topology_kind": "diamond_k4", "stage_cfg": {"pa": "A0", "pc1": "L1", "pc2": "R0", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S02_shift_left"},
            {"topology_kind": "diamond_k4", "stage_cfg": {"pa": "A1", "pc1": "L0", "pc2": "R1", "rd": "C0", "cross_coupled": True}, "tel": _tel(0.2, 0.2, 0.2), "event": "S03_coupled_first"},
            {"topology_kind": "diamond_k4", "stage_cfg": {"pa": "A1", "pc1": "L1", "pc2": "R0", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S04_comb_pl"},
            {"topology_kind": "diamond_k4", "stage_cfg": {"pa": "A0", "pc1": "L0", "pc2": "R1", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S05_rev_sep_break"},
            {"topology_kind": "diamond_k4", "stage_cfg": {"pa": "A0", "pc1": "L1", "pc2": "R1", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S06_comb_lr"},
            {"topology_kind": "diamond_k4", "stage_cfg": {"pa": "A1", "pc1": "L0", "pc2": "R1", "rd": "C0", "cross_coupled": True}, "tel": _tel(0.2, 0.2, 0.2), "event": "S07_revisit_coupled"},
            {"topology_kind": "diamond_k4", "stage_cfg": {"pa": "A0", "pc1": "L0", "pc2": "R1", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 0.2, 0.2), "event": "S08_revisit_sep"},
        ]
        return s0, steps

    if tc == "hybrid_mono_to_split_shift":
        # Stratum B HELD-OUT delayed-separability shift: Steps 0-1 monolithic, then shifts into K=3 cascade!
        s0 = {"topology_kind": "monolithic", "stage_cfg": {"core": "M0", "pa": "A0", "pb": "B0", "pc1": "L0", "pc2": "R0"}, "tel": _tel(0.2, 0.2, 0.2), "event": "S00_hyb_mono0"}
        steps = [
            {"topology_kind": "monolithic", "stage_cfg": {"core": "M1", "pa": "A0", "pb": "B0", "pc1": "L0", "pc2": "R0"}, "tel": _tel(1.6, 0.2, 0.2), "event": "S01_hyb_mono1"},
            {"topology_kind": "chain_k3", "stage_cfg": {"pa": "A0", "pb": "B0", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 1.7, 0.2), "event": "S02_hyb_k3_base"},
            {"topology_kind": "chain_k3", "stage_cfg": {"pa": "A0", "pb": "B1", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 1.7, 0.2), "event": "S03_hyb_k3_mid"},
            {"topology_kind": "chain_k3", "stage_cfg": {"pa": "A1", "pb": "B0", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 1.7, 0.2), "event": "S04_hyb_k3_head"},
            {"topology_kind": "chain_k3", "stage_cfg": {"pa": "A1", "pb": "B1", "rd": "C0", "cross_coupled": False}, "tel": _tel(0.2, 1.7, 0.2), "event": "S05_hyb_comb_hm"},
            {"topology_kind": "chain_k3", "stage_cfg": {"pa": "A0", "pb": "B0", "rd": "C1", "cross_coupled": False}, "tel": _tel(0.2, 1.7, 1.8), "event": "S06_hyb_k3_tail"},
            {"topology_kind": "chain_k3", "stage_cfg": {"pa": "A1", "pb": "B1", "rd": "C1", "cross_coupled": False}, "tel": _tel(0.2, 1.7, 1.8), "event": "S07_hyb_comb_hmt"},
            {"topology_kind": "monolithic", "stage_cfg": {"core": "M1", "pa": "A0", "pb": "B0", "pc1": "L0", "pc2": "R0"}, "tel": _tel(1.6, 0.2, 0.2), "event": "S08_hyb_revisit_mono1"},
        ]
        return s0, steps

    raise ValueError(f"Unsupported topology_class `{tc}`")


def _build_m101_task(
    *,
    task_id: str,
    short_name: str,
    world_seed: int,
    pub_seed: int,
    priv_s1: int,
    priv_s2: int,
    topology_kind: str,
    stage_cfg: dict[str, Any],
    telemetry_vec: list[float],
) -> HoldoutOODTask:
    pub_call = (
        f"_gen_m101_study({pub_seed}, world_seed={world_seed}, topology_kind={topology_kind!r}, "
        f"stage_cfg={stage_cfg!r}, telemetry_vec={telemetry_vec!r})"
    )
    priv_cases = (
        f"({priv_s1}, *_gen_m101_study({priv_s1}, world_seed={world_seed}, topology_kind={topology_kind!r}, "
        f"stage_cfg={stage_cfg!r}, telemetry_vec={telemetry_vec!r})),\n"
        f"({priv_s2}, *_gen_m101_study({priv_s2}, world_seed={world_seed}, topology_kind={topology_kind!r}, "
        f"stage_cfg={stage_cfg!r}, telemetry_vec={telemetry_vec!r})),"
    )
    schema_py = """from __future__ import annotations
import copy, importlib.util, json, os, pickle, sys
import numpy as np

_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "batch_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open(os.path.join(_d, "public_batch.pkl"), "rb") as fh:
    st = pickle.load(fh)[0]["study"]
out = mod.solve_batch(copy.deepcopy(st))
n_te = len(st["test"]["x"])
k_t = len(st["eval_times"])
errs = []
if not isinstance(out, dict):
    errs.append("not_dict")
else:
    for k, shp in (("y_hat", (n_te,)), ("state_pred", (n_te, k_t)), ("diagnostics_vec", (4,))):
        if k not in out:
            errs.append(f"missing:{k}")
            continue
        a = np.asarray(out[k], dtype=float)
        if a.shape != shp or not np.all(np.isfinite(a)):
            errs.append(f"bad:{k}:{a.shape}")
    if not errs:
        yh = np.asarray(out["y_hat"], dtype=float)
        sp = np.asarray(out["state_pred"], dtype=float)
        dv = np.asarray(out["diagnostics_vec"], dtype=float)
        exp_dv = np.array([float(np.mean(yh)), float(np.std(yh)), float(np.mean(sp[:, -1])), float(np.std(sp[:, -1]))], dtype=float)
        if float(np.max(np.abs(dv - exp_dv))) > 1e-3:
            errs.append("diagnostics_vec_handoff_mismatch")
print(json.dumps({"schema_valid": not errs, "errors": errs}, allow_nan=False))
sys.exit(1 if errs else 0)
"""
    pub_setup = (
        "import json, math, os, pickle, sys\nimport numpy as np\n"
        + _M101_WORLD_GEN_PY
        + f"\npub_st, _unused_gt = {pub_call}\n"
        'with open("/holdout_task/public_batch.pkl", "wb") as fh:\n'
        '    pickle.dump([{"study": pub_st}], fh)\n'
        + f'open("/holdout_task/validate_public_schema.py", "w").write({schema_py!r})\n'
    )
    priv_ver = (
        "import json, math, os, sys, time\nimport numpy as np\nsys.path.insert(0, '/holdout_task')\n"
        "import batch_solver as cand_mod\n"
        + _M101_WORLD_GEN_PY
        + f"\ncases = [\n{priv_cases}\n]\n"
        + """
t0 = time.monotonic()
valid = True; max_y_mae = None; max_tr_mae = None; max_dv_mae = None; rows = []
for sd, st, gt in cases:
    out = cand_mod.solve_batch(st)
    if not isinstance(out, dict) or not {"y_hat", "state_pred", "diagnostics_vec"} <= set(out.keys()):
        valid = False; rows.append({"seed": sd, "error": "MISSING_KEYS"}); continue
    yh = np.asarray(out["y_hat"], dtype=float)
    sp = np.asarray(out["state_pred"], dtype=float)
    dv = np.asarray(out["diagnostics_vec"], dtype=float)
    if yh.shape != gt["y_clean"].shape or sp.shape != gt["traj_clean"].shape or dv.shape != (4,):
        valid = False; rows.append({"seed": sd, "error": "SHAPE_MISMATCH"}); continue
    if not (np.all(np.isfinite(yh)) and np.all(np.isfinite(sp)) and np.all(np.isfinite(dv))):
        valid = False; rows.append({"seed": sd, "error": "NON_FINITE"}); continue
    y_mae = float(np.mean(np.abs(yh - gt["y_clean"])))
    tr_mae = float(np.mean(np.abs(sp - gt["traj_clean"])))
    dv_mae = float(np.mean(np.abs(dv - gt["summary_clean"])))
    exp_dv = np.array([float(np.mean(yh)), float(np.std(yh)), float(np.mean(sp[:, -1])), float(np.std(sp[:, -1]))], dtype=float)
    handoff_err = float(np.max(np.abs(dv - exp_dv)))
    max_y_mae = y_mae if max_y_mae is None else max(max_y_mae, y_mae)
    max_tr_mae = tr_mae if max_tr_mae is None else max(max_tr_mae, tr_mae)
    max_dv_mae = dv_mae if max_dv_mae is None else max(max_dv_mae, dv_mae)
    if handoff_err > 1e-3:
        valid = False; rows.append({"seed": sd, "error": "HANDOFF_MISMATCH", "y_mae": round(y_mae, 6), "tr_mae": round(tr_mae, 6)}); continue
    rows.append({"seed": sd, "y_mae": round(y_mae, 6), "tr_mae": round(tr_mae, 6), "dv_mae": round(dv_mae, 6)})
el = time.monotonic() - t0
sp_ok = bool(valid and len(rows) == 2 and max_y_mae is not None and max_tr_mae is not None and max_dv_mae is not None and max_y_mae <= 0.045 and max_tr_mae <= 0.045 and max_dv_mae <= 0.055 and el <= 25.0)
print(json.dumps({"strict_pass": sp_ok, "reward": 1.0 if sp_ok else 0.0, "max_y_mae": (round(max_y_mae, 6) if max_y_mae is not None else None), "max_traj_mae": (round(max_tr_mae, 6) if max_tr_mae is not None else None), "max_diag_mae": (round(max_dv_mae, 6) if max_dv_mae is not None else None), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}, allow_nan=False))
"""
    )
    return HoldoutOODTask(
        task_id=task_id,
        short_name=short_name,
        wave="m10_1_bounded_unknown_cut_dag_discovery",
        scientific_domain="blind_multi_stage_scientific_computation_graph",
        wormhole_family="m101_blind_dag",
        declared_artifacts=("/holdout_task/batch_solver.py",),
        declared_callable_entrypoints=("solve_batch",),
        public_pickle_filename="public_batch.pkl",
        host_only_public_gt_json="",
        instruction_text=BLIND_M101_INSTRUCTION_TEXT,
        public_setup_py=pub_setup,
        private_verifier_py=priv_ver,
    )


def build_m101_world_tasks(
    spec: M101WorldSpec,
) -> tuple[HoldoutOODTask, tuple[HoldoutOODTask, ...], list[dict[str, Any]]]:
    """Construct Step 0 (`Stage S`) + 8 continual stream tasks for an `M101WorldSpec` (`0` topology hints in task text/IDs!)."""
    s0_cfg, step_cfgs = _schedule_for_topology(spec)
    sb = spec.seed_base
    w_seed = sb + 7

    s_task = _build_m101_task(
        task_id=f"{spec.world_id}_batch_00__blind_campaign",
        short_name=f"{spec.world_id}_batch_00",
        world_seed=w_seed,
        pub_seed=sb + 10,
        priv_s1=sb + 20,
        priv_s2=sb + 30,
        topology_kind=s0_cfg["topology_kind"],
        stage_cfg=s0_cfg["stage_cfg"],
        telemetry_vec=s0_cfg["tel"],
    )
    stream_tasks: list[HoldoutOODTask] = []
    manifest: list[dict[str, Any]] = []
    for idx, sc in enumerate(step_cfgs, start=1):
        tk = _build_m101_task(
            task_id=f"{spec.world_id}_batch_{idx:02d}__blind_campaign",
            short_name=f"{spec.world_id}_batch_{idx:02d}",
            world_seed=w_seed,
            pub_seed=sb + 100 * idx + 10,
            priv_s1=sb + 100 * idx + 20,
            priv_s2=sb + 100 * idx + 30,
            topology_kind=sc["topology_kind"],
            stage_cfg=sc["stage_cfg"],
            telemetry_vec=sc["tel"],
        )
        stream_tasks.append(tk)
        manifest.append(
            {
                "step": idx,
                "task_id": tk.task_id,
                "short_name": tk.short_name,
                "event": sc["event"],
                "causal_event": sc["event"],
                "topology_kind": sc["topology_kind"],
                "stage_cfg": deepcopy(sc["stage_cfg"]),
                "telemetry_vec": list(sc["tel"]),
            }
        )
    return s_task, tuple(stream_tasks), manifest


def materialize_m101_public_study_and_sha(task: HoldoutOODTask) -> tuple[dict[str, Any], str, str]:
    """Deterministically materialize `study` dict, `public_batch.pkl` SHA-256, and canonical numeric SHA-256 (`0` private truth!)."""
    import pickle

    with tempfile.TemporaryDirectory(prefix="m101_pub_") as td:
        td_p = Path(td)
        patched_setup = task.public_setup_py.replace("/holdout_task/", str(td_p) + "/")
        ns: dict[str, Any] = {}
        exec(compile(patched_setup, "<m101_pub_setup>", "exec"), ns)
        pkl_bytes = (td_p / "public_batch.pkl").read_bytes()
        pub_sha = hashlib.sha256(pkl_bytes).hexdigest()
        study = pickle.loads(pkl_bytes)[0]["study"]
        num_sha = compute_canonical_numeric_item_sha256(study)
        return study, pub_sha, num_sha


# ============================================================================
# 3. Air-Gapped Docker Public Gate (Held-Out 56/16 Public Train Split + DAG Trace!)
# ============================================================================


def execute_m101_public_gate_in_container(
    *,
    task: HoldoutOODTask,
    candidate_code: str,
    expect_dag_trace: bool = False,
    y_mae_tol: float = 0.035,
    traj_mae_tol: float = 0.035,
    timeout_sec: int = 40,
) -> dict[str, Any]:
    """Execute candidate `batch_solver.py` inside isolated `hegel-clean-scipy:m7v1` container using ONLY public data:
      1. Runs `validate_public_schema.py` on `public_batch.pkl` (verifying shapes, finiteness, and `diagnostics_vec` handoff).
      2. Runs a pure-public held-out split (`train[:56]` -> predict `train[56:]` as `test`) verifying genuine scientific
         generalization (`y_mae <= y_mae_tol`, `traj_mae <= traj_mae_tol`) with ZERO private ground truth!
      3. When `expect_dag_trace=True`, captures `_dag_runtime_trace` and strictly enforces `all_exec_count_eq_1 == True`,
         `1 <= active_node_count <= 4`, `depth <= 4`, `max_fanout <= 2`, and non-identity intermediate nodes!
    """
    t0 = monotonic()
    cand_sha = hashlib.sha256(candidate_code.encode("utf-8")).hexdigest()
    setup_sha = hashlib.sha256(task.public_setup_py.encode("utf-8")).hexdigest()
    cid = m6f.start_detached_container(CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="2048m", cpus="2", network_mode="none")
    try:
        with tempfile.TemporaryDirectory(prefix="m101_gate_") as td:
            td_p = Path(td)
            (td_p / "setup.py").write_text(task.public_setup_py, encoding="utf-8")
            (td_p / "cand.py").write_text(candidate_code, encoding="utf-8")
            probe_py = f"""from __future__ import annotations
import copy, hashlib, importlib.util, json, os, pickle, sys
import numpy as np

os.environ["HEGEL_M101_DAG_TRACE_PATH"] = "/tmp/m101_dag_trace.json"
with open("/holdout_task/public_batch.pkl", "rb") as fh:
    raw_pkl = fh.read()
pub_sha = hashlib.sha256(raw_pkl).hexdigest()
st = pickle.loads(raw_pkl)[0]["study"]

spec = importlib.util.spec_from_file_location("_cand", "/holdout_task/batch_solver.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# 1. Full public test schema + handoff + DAG trace
out_full = mod.solve_batch(copy.deepcopy(st))
trace_full = None
if os.path.isfile("/tmp/m101_dag_trace.json"):
    trace_full = json.loads(open("/tmp/m101_dag_trace.json", "r", encoding="utf-8").read())

n_te = len(st["test"]["x"])
k_t = len(st["eval_times"])
errs = []
if not isinstance(out_full, dict):
    errs.append("not_dict")
else:
    for k, shp in (("y_hat", (n_te,)), ("state_pred", (n_te, k_t)), ("diagnostics_vec", (4,))):
        if k not in out_full:
            errs.append(f"missing:{{k}}"); continue
        a = np.asarray(out_full[k], dtype=float)
        if a.shape != shp or not np.all(np.isfinite(a)):
            errs.append(f"bad:{{k}}:{{a.shape}}")
    if not errs:
        yh = np.asarray(out_full["y_hat"], dtype=float)
        sp = np.asarray(out_full["state_pred"], dtype=float)
        dv = np.asarray(out_full["diagnostics_vec"], dtype=float)
        exp_dv = np.array([float(np.mean(yh)), float(np.std(yh)), float(np.mean(sp[:, -1])), float(np.std(sp[:, -1]))], dtype=float)
        if float(np.max(np.abs(dv - exp_dv))) > 1e-3:
            errs.append("diagnostics_vec_handoff_mismatch")

schema_ok = len(errs) == 0
cv_metrics = {{}}
if schema_ok:
    # 2. Pure-public held-out 52/20 train generalization check (`0` private oracle!)
    st_cv = copy.deepcopy(st)
    n_tr = len(st["train"]["x"])
    split_idx = max(40, n_tr - 20)
    val_x = st["train"]["x"][split_idx:].copy()
    val_y = st["train"]["y"][split_idx:].copy()
    val_tr = st["train"]["traj"][split_idx:].copy()
    st_cv["train"] = {{
        "x": st["train"]["x"][:split_idx].copy(),
        "y": st["train"]["y"][:split_idx].copy(),
        "traj": st["train"]["traj"][:split_idx].copy(),
        "summary": np.array([
            float(np.mean(st["train"]["y"][:split_idx])),
            float(np.std(st["train"]["y"][:split_idx])),
            float(np.mean(st["train"]["traj"][:split_idx, -1])),
            float(np.std(st["train"]["traj"][:split_idx, -1])),
        ], dtype=float),
    }}
    st_cv["test"] = {{"x": val_x}}
    out_cv = mod.solve_batch(st_cv)
    cv_y = np.asarray(out_cv["y_hat"], dtype=float)
    cv_tr = np.asarray(out_cv["state_pred"], dtype=float)
    if cv_y.shape != val_y.shape or cv_tr.shape != val_tr.shape or not (np.all(np.isfinite(cv_y)) and np.all(np.isfinite(cv_tr))):
        errs.append("cv_shape_or_nonfinite")
    else:
        y_mae = float(np.mean(np.abs(cv_y - val_y)))
        tr_mae = float(np.mean(np.abs(cv_tr - val_tr)))
        cv_metrics = {{"public_cv_y_mae": round(y_mae, 6), "public_cv_traj_mae": round(tr_mae, 6)}}
        if y_mae > {float(y_mae_tol)}:
            errs.append(f"public_cv_y_mae_exceeded:{{y_mae:.6f}}")
        if tr_mae > {float(traj_mae_tol)}:
            errs.append(f"public_cv_traj_mae_exceeded:{{tr_mae:.6f}}")

if {bool(expect_dag_trace)}:
    if not isinstance(trace_full, dict):
        errs.append("missing_dag_runtime_trace")
    else:
        if not trace_full.get("all_exec_count_eq_1"):
            errs.append("dag_shared_node_exec_count_ne_1")
        if not (1 <= int(trace_full.get("active_node_count", 0)) <= {M101_MAX_ACTIVE_NODES}):
            errs.append("dag_node_count_out_of_bounds")
        if int(trace_full.get("depth", 99)) > {M101_MAX_DEPTH}:
            errs.append("dag_depth_out_of_bounds")
        if int(trace_full.get("max_fanout", 99)) > {M101_MAX_FANOUT}:
            errs.append("dag_fanout_out_of_bounds")

print(json.dumps({{
    "schema_valid": schema_ok,
    "accepted": len(errs) == 0,
    "errors": errs,
    "public_input_sha256": pub_sha,
    "cv_metrics": cv_metrics,
    "dag_runtime_trace": trace_full,
    "kernel_version": {SHARED_KERNEL_VERSION!r},
}}, allow_nan=False))
"""
            (td_p / "probe.py").write_text(probe_py, encoding="utf-8")
            m6f.exec_container_bash(cid, "mkdir -p /holdout_task", timeout_sec=15)
            subprocess.run(["docker", "cp", str(td_p / "setup.py"), f"{cid}:/tmp/setup.py"], check=True, timeout=15)
            subprocess.run(["docker", "cp", str(td_p / "cand.py"), f"{cid}:/holdout_task/batch_solver.py"], check=True, timeout=15)
            subprocess.run(["docker", "cp", str(td_p / "probe.py"), f"{cid}:/tmp/probe.py"], check=True, timeout=15)
        setup_r = m6f.exec_container_bash(cid, "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/setup.py", timeout_sec=30)
        if setup_r["exit_code"] != 0:
            return {
                "accepted": False,
                "schema_valid": False,
                "execution_status": "infrastructure_error",
                "failure_kind": "timeout_or_infra",
                "errors": [f"setup_failed:{setup_r.get('stderr', '')[-200:]}"],
                "candidate_raw_sha256": cand_sha,
                "setup_source_sha256": setup_sha,
                "wall_ms": round((monotonic() - t0) * 1000.0, 2),
            }
        pr = m6f.exec_container_bash(
            cid, "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/probe.py", timeout_sec=timeout_sec
        )
        wall_ms = round((monotonic() - t0) * 1000.0, 2)
        if pr.get("timed_out") or pr["exit_code"] == 124:
            return {
                "accepted": False,
                "schema_valid": False,
                "execution_status": "timeout",
                "failure_kind": "timeout_or_infra",
                "errors": ["probe_timeout"],
                "candidate_raw_sha256": cand_sha,
                "setup_source_sha256": setup_sha,
                "wall_ms": wall_ms,
            }
        lines = [ln.strip() for ln in (pr.get("stdout") or "").splitlines() if ln.strip()]
        if pr["exit_code"] != 0 or not lines:
            err_tail = (pr.get("stderr") or pr.get("stdout") or "").strip()[-280:]
            is_shell_or_dag = "anti-identity-shell" in err_tail or "executed" in err_tail
            return {
                "accepted": False,
                "schema_valid": False,
                "execution_status": "completed",
                "failure_kind": ("schema_or_interface" if not is_shell_or_dag else "dag_invariant_violation"),
                "errors": [err_tail or f"rc={pr['exit_code']}"],
                "candidate_raw_sha256": cand_sha,
                "setup_source_sha256": setup_sha,
                "wall_ms": wall_ms,
            }
        parsed = json.loads(lines[-1])
        ok = bool(parsed.get("accepted"))
        fk = "none" if ok else ("schema_or_interface" if not parsed.get("schema_valid") else "scientific_or_numerical")
        return {
            "accepted": ok,
            "schema_valid": bool(parsed.get("schema_valid")),
            "execution_status": "completed",
            "failure_kind": fk,
            "errors": list(parsed.get("errors") or []),
            "public_input_sha256": str(parsed.get("public_input_sha256") or ""),
            "candidate_raw_sha256": cand_sha,
            "setup_source_sha256": setup_sha,
            "cv_metrics": dict(parsed.get("cv_metrics") or {}),
            "dag_runtime_trace": parsed.get("dag_runtime_trace"),
            "kernel_version": str(parsed.get("kernel_version") or SHARED_KERNEL_VERSION),
            "wall_ms": wall_ms,
        }
    finally:
        m6f.stop_container_safely(cid)


def grade_m101_candidate_in_container(*, task: HoldoutOODTask, candidate_code: str, timeout_sec: int = 45) -> dict[str, Any]:
    """Official private-verifier Docker grading for `M10.1` (runs 2 fresh unseen private seeds)."""
    t0 = monotonic()
    cid = m6f.start_detached_container(CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="2048m", cpus="2", network_mode="none")
    try:
        with tempfile.TemporaryDirectory(prefix="m101_priv_") as td:
            td_p = Path(td)
            (td_p / "cand.py").write_text(candidate_code, encoding="utf-8")
            (td_p / "ver.py").write_text(task.private_verifier_py, encoding="utf-8")
            m6f.exec_container_bash(cid, "mkdir -p /holdout_task", timeout_sec=15)
            subprocess.run(["docker", "cp", str(td_p / "cand.py"), f"{cid}:/holdout_task/batch_solver.py"], check=True, timeout=15)
            subprocess.run(["docker", "cp", str(td_p / "ver.py"), f"{cid}:/tmp/ver.py"], check=True, timeout=15)
        r = m6f.exec_container_bash(
            cid, "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/ver.py", timeout_sec=timeout_sec
        )
        wall_ms = round((monotonic() - t0) * 1000.0, 2)
        lines = [ln.strip() for ln in (r.get("stdout") or "").splitlines() if ln.strip()]
        if r["exit_code"] != 0 or not lines:
            return {
                "strict_pass": False,
                "reward": 0.0,
                "error": (r.get("stderr") or r.get("stdout") or f"rc={r['exit_code']}")[-300:],
                "verifier_wall_ms": wall_ms,
            }
        parsed = json.loads(lines[-1])
        parsed["verifier_wall_ms"] = wall_ms
        return parsed
    finally:
        m6f.stop_container_safely(cid)


def m101_task_to_frontier_task(
    task: HoldoutOODTask,
    *,
    continual_context_block: str = "",
) -> m6f.FrontierBenchmarkTask:
    instr = (continual_context_block + "\n\n" + task.instruction_text) if continual_context_block else task.instruction_text
    return m6f.FrontierBenchmarkTask(
        task_id=task.task_id,
        short_name=task.short_name,
        benchmark="Hegel_M10_1_UnknownCut_DAG_Discovery",
        wave=task.wave,
        scientific_domain=task.scientific_domain,
        wormhole_family=task.wormhole_family,
        instruction_text=instr,
        agent_image_tag=CLEAN_SCIPY_SANDBOX_IMAGE,
        verifier_image_tag=CLEAN_SCIPY_SANDBOX_IMAGE,
        declared_artifacts=task.declared_artifacts,
        declared_callable_entrypoints=task.declared_callable_entrypoints,
        required_public_assets=("/holdout_task/public_batch.pkl", "/holdout_task/validate_public_schema.py"),
        holdout_public_setup_py=task.public_setup_py,
        host_only_public_gt_json="",
        verifier_timeout_sec=180,
        agent_cpus=2,
        agent_memory_mb=2048,
        verifier_cpus=2,
        verifier_memory_mb=2048,
    )


# ============================================================================
# 4. Reference `0`-LLM Dev DAG Nodes & Solvers (Gate 0–2 Unit/Docker Tests ONLY!)
# ============================================================================

def build_ref_dev_node_pa_code(mode: str) -> str:
    """Mode-specific reference node for `panel_a` inverse (`A0`, `A1`, `A2`) — Gate 0–2 unit tests ONLY!"""
    if mode == "A0":
        return r'''from __future__ import annotations
import numpy as np

def run_node(inputs: dict) -> dict:
    study = inputs["__input__"]
    u = np.asarray(study["panel_a"]["u"], dtype=float)
    v = np.asarray(study["panel_a"]["v"], dtype=float)
    aug = np.column_stack([u, np.ones(len(u))])
    w = np.ones(len(u))
    for _ in range(8):
        sw = np.sqrt(w)[:, None]
        W, *_ = np.linalg.lstsq(aug * sw, v * sw, rcond=None)
        res = np.linalg.norm(v - aug @ W, axis=1)
        w = 1.0 / (1.0 + (res / max(2.5 * float(np.median(res)), 1e-4)) ** 2)
    res = np.linalg.norm(v - aug @ W, axis=1)
    keep = res <= max(5.0 * float(np.median(res)), 0.05)
    W, *_ = np.linalg.lstsq(aug[keep], v[keep], rcond=None)
    A_inv = np.linalg.inv(W[:4, :])
    b_fwd = W[4, :]
    tr_x = np.asarray(study["train"]["x"], dtype=float)
    te_x = np.asarray(study["test"]["x"], dtype=float)
    return {
        "z_tr": (tr_x - b_fwd[None, :]) @ A_inv,
        "z_te": (te_x - b_fwd[None, :]) @ A_inv,
    }
'''
    return r'''from __future__ import annotations
import numpy as np

def run_node(inputs: dict) -> dict:
    study = inputs["__input__"]
    u = np.asarray(study["panel_a"]["u"], dtype=float)
    v = np.asarray(study["panel_a"]["v"], dtype=float)
    def _fwd_phi(z):
        zr = np.roll(z, -1, axis=1)
        return np.column_stack([np.ones(len(z)), z, zr, z * zr, z ** 3])
    # Robust outlier filter
    P = _fwd_phi(u)
    w = np.ones(len(u))
    for _ in range(6):
        sw = np.sqrt(w)[:, None]
        W, *_ = np.linalg.lstsq(P * sw, v * sw, rcond=None)
        res = np.linalg.norm(v - P @ W, axis=1)
        w = 1.0 / (1.0 + (res / max(3.0 * float(np.median(res)), 1e-4)) ** 2)
    keep = np.linalg.norm(v - P @ W, axis=1) <= max(5.0 * float(np.median(np.linalg.norm(v - P @ W, axis=1))), 0.1)
    W, *_ = np.linalg.lstsq(P[keep], v[keep], rcond=None)
    # Invert forward map by damped Newton iteration
    def _inv(x_obs):
        z = np.zeros_like(x_obs)
        eps = 1e-5
        for _ in range(12):
            f = _fwd_phi(z) @ W - x_obs
            J = np.zeros((len(z), 4, 4), dtype=float)
            for j in range(4):
                zp = z.copy(); zp[:, j] += eps
                J[:, :, j] = (_fwd_phi(zp) @ W - _fwd_phi(z) @ W) / eps
            dz = np.linalg.solve(J, f[:, :, None])[:, :, 0]
            z = z - dz
            if float(np.max(np.abs(dz))) < 1e-9:
                break
        return z
    return {"z_tr": _inv(np.asarray(study["train"]["x"], dtype=float)), "z_te": _inv(np.asarray(study["test"]["x"], dtype=float))}
'''


def build_ref_dev_node_pb_code(mode: str) -> str:
    """Mode-specific reference node for `panel_b` (`B0`, `B1`) — Gate 0–2 unit tests ONLY!"""
    if mode == "B0":
        expr = "x"
    elif mode == "B1":
        expr = "np.sign(x) * (np.maximum(np.abs(x) - 0.35, 0.0) ** 1.5)"
    else:
        expr = "np.cos(2.6 * x)"
    return f'''from __future__ import annotations
import numpy as np

def run_node(inputs: dict) -> dict:
    study = inputs["__input__"]
    prev = [v for k, v in inputs.items() if k != "__input__"][0]
    u = np.asarray(study["panel_b"]["u"], dtype=float)
    v = np.asarray(study["panel_b"]["v"], dtype=float)
    def _act(x):
        return {expr}
    P = np.column_stack([_act(u), np.ones(len(u))])
    W, *_ = np.linalg.lstsq(P, v, rcond=None)
    return {{
        "h_tr": np.column_stack([_act(prev["z_tr"]), np.ones(len(prev["z_tr"]))]) @ W,
        "h_te": np.column_stack([_act(prev["z_te"]), np.ones(len(prev["z_te"]))]) @ W,
    }}
'''


def build_ref_dev_node_pc_code(target_key: str, mode: str) -> str:
    """Mode-specific reference node for `panel_c` (`L0/L1/R0/R1`) — Gate 0–2 unit tests ONLY!"""
    if mode in ("L0", "H0"):
        expr = "np.sign(x) * (1.0 - np.exp(-1.8 * np.abs(x)))"
    elif mode in ("L1", "H1"):
        expr = "x * np.roll(x, -1, axis=1)"
    elif mode == "R0":
        expr = "np.log1p(np.exp(2.2 * x)) / 2.2 - 0.315"
    else:
        expr = "np.cos(2.4 * x)"
    return f'''from __future__ import annotations
import numpy as np

def run_node(inputs: dict) -> dict:
    study = inputs["__input__"]
    prev = [v for k, v in inputs.items() if k != "__input__"][0]
    u = np.asarray(study["panel_c"]["u"], dtype=float)
    v = np.asarray(study["panel_c"][{target_key!r}], dtype=float)
    z_tr = prev["z_tr"] if "z_tr" in prev else prev["h_tr"]
    z_te = prev["z_te"] if "z_te" in prev else prev["h_te"]
    def _act(x):
        return {expr}
    P = np.column_stack([_act(u), np.ones(len(u))])
    W, *_ = np.linalg.lstsq(P, v, rcond=None)
    return {{
        "b_tr": np.column_stack([_act(z_tr), np.ones(len(z_tr))]) @ W,
        "b_te": np.column_stack([_act(z_te), np.ones(len(z_te))]) @ W,
    }}
'''


REF_DEV_NODE_JOIN_READOUT_PY = r'''from __future__ import annotations
import numpy as np

def run_node(inputs: dict) -> dict:
    study = inputs["__input__"]
    tr_y = np.asarray(study["train"]["y"], dtype=float)
    tr_traj = np.asarray(study["train"]["traj"], dtype=float)
    feats_tr = []
    feats_te = []
    for k in sorted(inputs.keys()):
        if k == "__input__":
            continue
        d = inputs[k]
        for fk_tr, fk_te in (("b_tr", "b_te"), ("h_tr", "h_te"), ("z_tr", "z_te")):
            if fk_tr in d and fk_te in d:
                feats_tr.append(np.asarray(d[fk_tr], dtype=float))
                feats_te.append(np.asarray(d[fk_te], dtype=float))
                break
    def _design(flist):
        cols = [np.ones(len(flist[0]))]
        for a in flist:
            cols.append(a)
            cols.append(np.sum(a[:, :2] * a[:, 2:], axis=1)[:, None])
        if len(flist) >= 2:
            a, b = flist[0], flist[1]
            cols.append(np.sum(a * b, axis=1)[:, None])
            cols.append((a[:, 0] * b[:, 1])[:, None])
            cols.append((b[:, 0] * a[:, 1])[:, None])
        return np.column_stack(cols)
    X_tr = _design(feats_tr)
    X_te = _design(feats_te)
    reg = 1e-6 * np.eye(X_tr.shape[1])
    w_y = np.linalg.solve(X_tr.T @ X_tr + reg, X_tr.T @ tr_y)
    w_tr = np.linalg.solve(X_tr.T @ X_tr + reg, X_tr.T @ tr_traj)
    yh = X_te @ w_y
    sp = X_te @ w_tr
    dv = np.array([float(np.mean(yh)), float(np.std(yh)), float(np.mean(sp[:, -1])), float(np.std(sp[:, -1]))], dtype=float)
    return {"y_hat": yh, "state_pred": sp, "diagnostics_vec": dv}
'''


__all__ = [
    "BLIND_M101_INSTRUCTION_TEXT",
    "CONFIRMATION_M101_WORLD_SPECS",
    "CutWithdrawalMergeReceipt",
    "DEV_M101_WORLD_SPECS",
    "ExecutableComputationDAG",
    "ExecutableModuleNode",
    "M101WorldSpec",
    "M101_ARMS",
    "M101_DAG_LEDGER_FOREST_KEY",
    "M101_LEGAL_DAG_ACTIONS",
    "M101_MAX_ACTIVE_NODES",
    "M101_MAX_DEPTH",
    "M101_MAX_FANOUT",
    "M101_OBSERVABLE_NAMES",
    "M101_OUTPUT_KEYS",
    "REF_DEV_NODE_JOIN_READOUT_PY",
    "build_m101_world_tasks",
    "build_ref_dev_node_pa_code",
    "build_ref_dev_node_pb_code",
    "build_ref_dev_node_pc_code",
    "compile_m101_dag_to_instrumented_solver_code",
    "compute_m101_neutral_observables",
    "execute_m101_public_gate_in_container",
    "grade_m101_candidate_in_container",
    "m101_task_to_frontier_task",
    "materialize_m101_public_study_and_sha",
    "verify_m101_cut_withdrawal_external_fk",
]
