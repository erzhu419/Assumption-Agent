r"""HippoRAG Specificity-Weighted Personalized PageRank & Equal-Metadata Flat Retrievers (Hegel V17.0 M3).

Implements neurobiologically inspired associative graph recall over ``PhilosophicalForest``
(Gutiérrez et al., NeurIPS 2024 HippoRAG; 2025 HippoRAG 2) with strict scientific boundaries:

1. **Zero-Leakage Observable Indexing**: Signature cues and query seeds consist strictly
   of empirically observed ``sig:*`` protocol/pilot signatures — never domain names, feature
   names, or ground-truth mechanism labels.
2. **Concept-Forest + Horizontal-Wormhole Markov Topology**:
   - Specificity-weighted cue seeding: each observed cue $\sigma \in Q$ carries HippoRAG
     node-specificity weight $s(\sigma) = 1 / \deg(\sigma)^{1.15}$, distributed across the
     $\deg(\sigma)$ concept nodes exhibiting $\sigma$ so broad hub cues never transit-leak
     mass across unrelated trees during random walks.
   - Vertical intra-tree transitions: ``ParentNode -> RefinedChildNode`` (dialectical
     escalation along ``local_implication_edges``) with weak ``Child -> Parent`` backoff.
   - Horizontal cross-tree wormhole transitions: ``MidNode_A <-> MidNode_B`` (live, non-stale
     ``HorizontalWormholeEdge``s weighted inversely by empirical ``hop_error_budget``).
3. **Probability-Conserving Equivalence-Class PPR**: Stationary distribution $\mathbf{p}$
   satisfies $\sum_v p(v) = 1$; corroborated cross-tree wormhole twins sharing a live
   ``HorizontalWormholeEdge`` pool their conserved Markov class probability.
4. **Recall != Authorization**: PPR and Flat retrievers ONLY produce Tier-1 ``retrieve_candidate``
   rankings along with the associated wormhole path metadata; downstream ``compose_wormhole_chain``,
   ``authorize_abstraction_use``, and prefix-isolated target confirmation remain mandatory and
   identical across all retrieval arms!
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import hashlib
import json
import math
import time
from typing import Any

from .wormhole_forest import PhilosophicalForest


_FORBIDDEN_QUERY_PREFIXES = ("schema:", "mech:", "domain:", "task:", "truth:", "oracle:")


def _node_content_tie_key(node: dict[str, Any]) -> str:
    """Identifier-independent deterministic tie-breaker over canonical contract & observable cues."""
    cc = node.get("claim_contract", {})
    payload = {
        "action": cc.get("conclusion_or_action", {}),
        "preconditions": cc.get("preconditions", ()),
        "sigs": sorted(node.get("probe_signatures", ())),
        "sign": node.get("observation_map", {}).get("sign_convention", 1),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _build_same_schema_wormhole_components(
    node_ids: list[str],
    active_nodes: dict[str, dict[str, Any]],
    adj_wh: dict[str, list[dict[str, Any]]],
) -> dict[str, list[str]]:
    """Connected components of live, non-stale same-schema HorizontalWormholeEdges."""
    same_schema_adj: dict[str, list[str]] = {nid: [] for nid in node_ids}
    seen_wids: set[str] = set()
    for wh_list in adj_wh.values():
        for wh in wh_list:
            wid = wh["wormhole_id"]
            if wid in seen_wids:
                continue
            seen_wids.add(wid)
            na, nb = wh["node_a_id"], wh["node_b_id"]
            if na in active_nodes and nb in active_nodes:
                act_a = active_nodes[na]["claim_contract"]["conclusion_or_action"]
                act_b = active_nodes[nb]["claim_contract"]["conclusion_or_action"]
                sa_tag = act_a.get("schema_tag") or act_a.get("ast_skeleton_digest")
                sb_tag = act_b.get("schema_tag") or act_b.get("ast_skeleton_digest")
                if sa_tag and sa_tag == sb_tag:
                    same_schema_adj[na].append(nb)
                    same_schema_adj[nb].append(na)

    wh_comp_of: dict[str, list[str]] = {}
    visited: set[str] = set()
    for nid in node_ids:
        if nid in visited:
            continue
        comp: list[str] = []
        q: deque[str] = deque([nid])
        visited.add(nid)
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nxt in same_schema_adj.get(cur, ()):
                if nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        comp.sort()
        for m in comp:
            wh_comp_of[m] = comp
    return wh_comp_of


def validate_observable_query_signatures(signatures: list[str]) -> list[str]:
    """Ensure retrieval queries contain ONLY observable ``sig:*`` tokens (zero label leakage)."""
    cleaned: list[str] = []
    for tok in signatures:
        s = str(tok).strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith(_FORBIDDEN_QUERY_PREFIXES) or not s.startswith("sig:"):
            raise ValueError(f"Forbidden non-observable token in retrieval query: {s!r}")
        if s not in cleaned:
            cleaned.append(s)
    return cleaned


def _extract_node_incident_metadata(
    forest: PhilosophicalForest,
) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, str], dict[str, Any]]]:
    """Collect live non-stale horizontal wormhole adjacency for path & metadata parity."""
    adj_wh: dict[str, list[dict[str, Any]]] = {}
    pair_wh: dict[tuple[str, str], dict[str, Any]] = {}
    for wh in forest.horizontal_wormhole_edges.values():
        if wh.get("stale_endpoint_revision"):
            continue
        na, nb = wh["node_a_id"], wh["node_b_id"]
        if na not in forest.nodes or nb not in forest.nodes:
            continue
        if wh.get("node_a_version") != forest.nodes[na]["version"]:
            continue
        if wh.get("node_b_version") != forest.nodes[nb]["version"]:
            continue
        adj_wh.setdefault(na, []).append(wh)
        adj_wh.setdefault(nb, []).append(wh)
        pair_wh[(na, nb)] = wh
        pair_wh[(nb, na)] = wh
    return adj_wh, pair_wh


class HippoRAGWormholeIndex:
    """Specificity-weighted Personalized PageRank index over ``PhilosophicalForest``."""

    def __init__(
        self,
        forest: PhilosophicalForest,
        *,
        allow_horizontal_wormholes: bool = True,
        wormhole_equivalence_pooling: bool = True,
        include_coarse_graining_edges: bool = True,
        scale_aware: bool = True,
        damping_alpha: float = 0.80,
        specificity_exponent: float = 1.15,
        self_loop_weight: float = 0.25,
        refined_sink_self_weight: float = 2.40,
        vertical_up_weight: float = 1.85,
        vertical_down_weight: float = 0.10,
        coarse_up_weight: float = 1.15,
        fine_down_weight: float = 0.85,
        wormhole_base_weight: float = 1.55,
        max_ppr_iterations: int = 50,
        convergence_tol: float = 1e-9,
    ) -> None:
        self._forest_ref = forest
        self.allow_horizontal_wormholes = bool(allow_horizontal_wormholes)
        self.wormhole_equivalence_pooling = bool(wormhole_equivalence_pooling)
        self.include_coarse_graining_edges = bool(include_coarse_graining_edges)
        self.scale_aware = bool(scale_aware)
        self.damping_alpha = float(damping_alpha)
        self.specificity_exponent = float(specificity_exponent)
        self._self_loop_weight = float(self_loop_weight)
        self._refined_sink_self_weight = float(refined_sink_self_weight)
        self._vertical_up_weight = float(vertical_up_weight)
        self._vertical_down_weight = float(vertical_down_weight)
        self._coarse_up_weight = float(coarse_up_weight)
        self._fine_down_weight = float(fine_down_weight)
        self._wormhole_base_weight = float(wormhole_base_weight)
        self.max_ppr_iterations = int(max_ppr_iterations)
        self.convergence_tol = float(convergence_tol)
        self.refresh_count: int = 0
        self._indexed_forest_state: tuple[Any, ...] = ()
        self._build_from_forest(forest)

    def _build_from_forest(self, forest: PhilosophicalForest) -> None:
        t0 = time.perf_counter()
        self.active_nodes: dict[str, dict[str, Any]] = {}
        for nid, node in sorted(forest.nodes.items()):
            if node["utility_ledger"].get("archived_low_utility"):
                continue
            if node["claim_contract"].get("evidence_status") != "SUPPORTED_IN_SCOPE":
                continue
            self.active_nodes[nid] = deepcopy(node)

        self.adj_wh, self.pair_wh = _extract_node_incident_metadata(forest)

        # Observable signature inverted index (`sig:* -> [nid, ...]`)
        self.sig_to_nodes: dict[str, list[str]] = {}
        n_obs_edges = 0
        for nid, node in self.active_nodes.items():
            for sig in validate_observable_query_signatures(list(node.get("probe_signatures", ()))):
                self.sig_to_nodes.setdefault(sig, []).append(nid)
                n_obs_edges += 1

        self.node_ids: list[str] = sorted(self.active_nodes.keys())
        self.n_to_idx: dict[str, int] = {nid: i for i, nid in enumerate(self.node_ids)}
        self.content_tie_keys: dict[str, str] = {
            nid: _node_content_tie_key(node) for nid, node in self.active_nodes.items()
        }
        n_verts = len(self.node_ids)
        adj_weights: list[dict[int, float]] = [{} for _ in range(n_verts)]

        def _add_directed(ua: str, vb: str, w: float) -> None:
            if ua not in self.n_to_idx or vb not in self.n_to_idx:
                return
            ia, ib = self.n_to_idx[ua], self.n_to_idx[vb]
            adj_weights[ia][ib] = adj_weights[ia].get(ib, 0.0) + float(w)

        refined_children: set[str] = set()
        parents_with_children: set[str] = set()
        n_vert_edges = 0
        self.graph_adj_nodes: dict[str, list[tuple[str, dict[str, Any] | None]]] = {
            nid: [] for nid in self.active_nodes
        }

        # 1. Intra-tree vertical implication / sublation edges (`parent -> stronger_child`)
        for v_edge in forest.local_implication_edges:
            if v_edge.get("stale_endpoint_revision"):
                continue
            sa, wa = v_edge["stronger_node_id"], v_edge["weaker_node_id"]
            if sa in self.active_nodes and wa in self.active_nodes:
                if (
                    v_edge["stronger_version"] == self.active_nodes[sa]["version"]
                    and v_edge["weaker_version"] == self.active_nodes[wa]["version"]
                ):
                    _add_directed(wa, sa, self._vertical_up_weight)
                    _add_directed(sa, wa, self._vertical_down_weight)
                    refined_children.add(sa)
                    parents_with_children.add(wa)
                    self.graph_adj_nodes[sa].append((wa, None))
                    self.graph_adj_nodes[wa].append((sa, None))
                    n_vert_edges += 1

        # 1b. Intra-tree multi-scale coarse-graining edges (`fine -> coarse` via Pi_s & reverse escalation)
        n_coarse_edges = 0
        if self.include_coarse_graining_edges:
            for cg_edge in getattr(forest, "coarse_graining_edges", ()):
                if cg_edge.get("stale_endpoint_revision"):
                    continue
                fa, ca = cg_edge["fine_node_id"], cg_edge["coarse_node_id"]
                if fa in self.active_nodes and ca in self.active_nodes:
                    if (
                        cg_edge["fine_version"] == self.active_nodes[fa]["version"]
                        and cg_edge["coarse_version"] == self.active_nodes[ca]["version"]
                    ):
                        _add_directed(fa, ca, self._coarse_up_weight)
                        _add_directed(ca, fa, self._fine_down_weight)
                        self.graph_adj_nodes[fa].append((ca, None))
                        self.graph_adj_nodes[ca].append((fa, None))
                        n_coarse_edges += 1

        # 2. Self-retention weights (higher on terminal witness-refined nodes so escalated mass concentrates)
        for nid in self.node_ids:
            if nid in refined_children and nid not in parents_with_children:
                _add_directed(nid, nid, self._refined_sink_self_weight)
            else:
                _add_directed(nid, nid, self._self_loop_weight)

        # 3. Cross-tree horizontal wormhole edges (toggled off in VerticalPPR ablation!)
        n_wh_edges = 0
        if self.allow_horizontal_wormholes:
            seen_wh_ids: set[str] = set()
            for wh_list in self.adj_wh.values():
                for wh in wh_list:
                    wid = wh["wormhole_id"]
                    if wid in seen_wh_ids:
                        continue
                    seen_wh_ids.add(wid)
                    na, nb = wh["node_a_id"], wh["node_b_id"]
                    if na in self.active_nodes and nb in self.active_nodes:
                        hop_err = max(0.0, float(wh.get("hop_error_budget", 0.02)))
                        w_wh = self._wormhole_base_weight / (1.0 + 8.0 * hop_err)
                        _add_directed(na, nb, w_wh)
                        _add_directed(nb, na, w_wh)
                        self.graph_adj_nodes[na].append((nb, wh))
                        self.graph_adj_nodes[nb].append((na, wh))
                        n_wh_edges += 1

        if self.allow_horizontal_wormholes and self.wormhole_equivalence_pooling:
            self.wh_component_of = _build_same_schema_wormhole_components(
                self.node_ids, self.active_nodes, self.adj_wh
            )
        else:
            self.wh_component_of = {nid: [nid] for nid in self.node_ids}

        # Row-stochastic transition matrix for probability-conserving power iteration
        self.singleton_bg_mass: list[float] = [0.0] * n_verts
        self.transitions: list[list[tuple[int, float]]] = []
        for u_idx in range(n_verts):
            nbrs = adj_weights[u_idx]
            tot = sum(nbrs.values())
            if tot <= 1e-15:
                self.transitions.append([(u_idx, 1.0)])
            else:
                self.transitions.append([(v_idx, w / tot) for v_idx, w in sorted(nbrs.items())])

        self.build_wall_seconds = time.perf_counter() - t0
        self.edge_counts = {
            "signature_cues": len(self.sig_to_nodes),
            "concept_node_vertices": len(self.active_nodes),
            "bipartite_obs_links": n_obs_edges,
            "vertical_tree_edges": n_vert_edges,
            "coarse_graining_edges": n_coarse_edges,
            "horizontal_wormhole_edges": n_wh_edges,
        }
        total_edges = n_obs_edges + n_vert_edges + n_coarse_edges + n_wh_edges
        self.index_build_compute_cost = round(0.35 * (len(self.sig_to_nodes) + n_verts) + 0.15 * total_edges, 4)
        self._indexed_forest_state = forest.forest_state_signature()

    def ensure_fresh(self) -> float:
        """Invalidate and refresh index whenever the underlying ``PhilosophicalForest`` was mutated."""
        cur_state = self._forest_ref.forest_state_signature()
        if cur_state == self._indexed_forest_state:
            return 0.0
        self._build_from_forest(self._forest_ref)
        self.refresh_count += 1
        return float(self.index_build_compute_cost)

    def _shortest_wormhole_path_from_direct_hits(
        self,
        target_nid: str,
        direct_hit_nids: set[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """Find shortest wormhole path connecting any directly signature-matched node to ``target_nid``."""
        primary_incident = (self.adj_wh.get(target_nid) or [None])[0]
        if target_nid in direct_hit_nids:
            if primary_incident is not None:
                p_wid = primary_incident.get("role_and_observation_map", {}).get("chain_parent_wormhole")
                if p_wid:
                    for wh_list in self.adj_wh.values():
                        for cand_wh in wh_list:
                            if cand_wh["wormhole_id"] == p_wid:
                                return [deepcopy(cand_wh), deepcopy(primary_incident)], deepcopy(primary_incident)
                return [deepcopy(primary_incident)], deepcopy(primary_incident)
            return [], None

        if not direct_hit_nids or not self.allow_horizontal_wormholes:
            return ([deepcopy(primary_incident)] if primary_incident else []), deepcopy(primary_incident)

        queue: deque[tuple[str, list[dict[str, Any]]]] = deque((src, []) for src in sorted(direct_hit_nids))
        visited = set(direct_hit_nids)
        while queue:
            cur, wh_chain = queue.popleft()
            if len(wh_chain) >= 3:
                continue
            for nxt, wh_edge in self.graph_adj_nodes.get(cur, ()):
                if nxt in visited:
                    continue
                visited.add(nxt)
                next_chain = [*wh_chain, deepcopy(wh_edge)] if wh_edge is not None else list(wh_chain)
                if nxt == target_nid:
                    rep_wh = next_chain[-1] if next_chain else primary_incident
                    return next_chain, deepcopy(rep_wh)
                queue.append((nxt, next_chain))

        return ([deepcopy(primary_incident)] if primary_incident else []), deepcopy(primary_incident)

    def retrieve(
        self,
        observed_signatures: list[str],
        *,
        task_view: dict[str, Any],
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Run specificity-weighted Personalized PageRank and return Top-K Tier-1 candidates."""
        t0 = time.perf_counter()
        refresh_cost = self.ensure_fresh()
        valid_sigs = [
            s for s in validate_observable_query_signatures(observed_signatures)
            if s in self.sig_to_nodes
        ]
        n_verts = len(self.node_ids)
        if not valid_sigs or n_verts == 0:
            return {
                "candidates": [],
                "iterations": 0,
                "stationary_mass_sum": 0.0,
                "non_direct_walk_recalled_count": 0,
                "query_compute_cost": round(1.0 + refresh_cost, 4),
                "index_refreshed": refresh_cost > 0.0,
                "index_refresh_compute_cost": refresh_cost,
                "query_wall_seconds": time.perf_counter() - t0,
                "seed_specificity_weights": {},
            }

        # HippoRAG node-specificity reset: cue weight s(sig) = 1 / deg(sig)^p distributed across incident nodes
        cue_spec = {s: 1.0 / (max(1, len(self.sig_to_nodes[s])) ** self.specificity_exponent) for s in valid_sigs}
        cue_sum = sum(cue_spec.values())
        reset_vec = [0.0] * n_verts
        seed_weights: dict[str, float] = {}
        for s, w in cue_spec.items():
            nw = w / cue_sum
            seed_weights[s] = round(nw, 6)
            inc_nodes = self.sig_to_nodes[s]
            share = nw / len(inc_nodes)
            for nid in inc_nodes:
                reset_vec[self.n_to_idx[nid]] += share

        p = list(reset_vec)
        alpha = self.damping_alpha
        iters = 0
        if alpha > 1e-12:
            for it in range(1, self.max_ppr_iterations + 1):
                iters = it
                nxt = [(1.0 - alpha) * r for r in reset_vec]
                bg_pool = 0.0
                for u_idx, pu in enumerate(p):
                    if pu <= 1e-15:
                        continue
                    scaled = alpha * pu
                    for v_idx, prob in self.transitions[u_idx]:
                        nxt[v_idx] += scaled * prob
                    if self.singleton_bg_mass[u_idx] > 0.0:
                        bg_pool += scaled * self.singleton_bg_mass[u_idx]
                if bg_pool > 0.0:
                    for v_idx, rv in enumerate(reset_vec):
                        if rv > 0.0:
                            nxt[v_idx] += bg_pool * rv
                diff = sum(abs(a - b) for a, b in zip(nxt, p))
                p = nxt
                if diff <= self.convergence_tol:
                    break

        direct_hit_nids = {nid for s in valid_sigs for nid in self.sig_to_nodes.get(s, ())}
        goal = task_view.get("goal_kind")
        req_arity = task_view.get("active_arity")
        req_scale = task_view.get("target_scale_regime") or task_view.get("scale_regime")

        scored_nodes: list[dict[str, Any]] = []
        for nid, node in self.active_nodes.items():
            node_p = p[self.n_to_idx[nid]]
            comp_members = self.wh_component_of.get(nid, [nid])
            class_p = sum(p[self.n_to_idx[m]] for m in comp_members)
            if class_p <= 1e-12:
                continue
            node_arity = node["observation_map"].get("parameter_semantics", {}).get("active_arity")
            if req_arity is not None and node_arity is not None and int(node_arity) != int(req_arity):
                continue
            if self.scale_aware and req_scale is not None:
                sm = node.get("scale_map") or {}
                node_scale = (sm.get("window_or_partition") or {}).get("scale_regime")
                if node_scale is not None and node_scale != req_scale:
                    class_p *= 0.05
                    node_p *= 0.05
            wh_chain, rep_wh = self._shortest_wormhole_path_from_direct_hits(nid, direct_hit_nids)
            if rep_wh is not None and goal and rep_wh.get("compatible_goal_kinds"):
                if goal not in rep_wh["compatible_goal_kinds"]:
                    continue
            if self.scale_aware and req_scale is not None and rep_wh is not None:
                wh_scale = rep_wh.get("scale_regime")
                if wh_scale is not None and wh_scale != req_scale:
                    class_p *= 0.20
            matched_sigs = sorted(set(valid_sigs) & set(node.get("probe_signatures", ())))
            scored_nodes.append({
                "node_id": nid,
                "target_node": deepcopy(node),
                "node_ppr": round(float(node_p), 8),
                "ppr_score": round(float(class_p), 8),
                "score": round(float(class_p), 8),
                "matched_signatures": matched_sigs,
                "reached_only_by_graph_walk": (nid not in direct_hit_nids),
                "wormhole_edge": rep_wh,
                "wormhole_path_edges": wh_chain,
                "entry_kind": ("horizontal_mid_wormhole" if wh_chain else "local_tree_node"),
                "granted_tier": "retrieve_candidate",
                "_tie_key": self.content_tie_keys[nid],
            })

        # Content-hash tie-breaking (strictly invariant to node_id / domain string renaming!)
        scored_nodes.sort(key=lambda item: (-item["ppr_score"], -item["node_ppr"], item["_tie_key"]))
        deduped: list[dict[str, Any]] = []
        seen_schemas: set[str] = set()
        for item in scored_nodes:
            st = str(item["target_node"]["claim_contract"]["conclusion_or_action"].get("schema_tag") or item["_tie_key"])
            if st in seen_schemas:
                continue
            seen_schemas.add(st)
            item_clean = {k: v for k, v in item.items() if k != "_tie_key"}
            deduped.append(item_clean)
            if len(deduped) >= top_k:
                break

        non_direct_walk_count = sum(1 for c in deduped if c.get("reached_only_by_graph_walk"))
        q_compute = round(1.4 + 0.025 * len(self.active_nodes) + 0.03 * iters + refresh_cost, 4)
        return {
            "candidates": deduped,
            "iterations": iters,
            "stationary_mass_sum": round(sum(p), 8),
            "non_direct_walk_recalled_count": non_direct_walk_count,
            "query_compute_cost": q_compute,
            "index_refreshed": refresh_cost > 0.0,
            "index_refresh_compute_cost": refresh_cost,
            "query_wall_seconds": time.perf_counter() - t0,
            "seed_specificity_weights": seed_weights,
        }


class FlatEqualMetadataIndex:
    """Strict Equal-Content Flat Retriever (BM25 or IDF-Cosine) over the EXACT SAME Forest Metadata!

    Preserves 100% of every node's ``ClaimContract``, ``ObservationMap``, ``ScaleMap``,
    ``ALIGN`` exclusions, and cumulative ``wormhole_path_edges``; supports optional
    ``wormhole_equivalence_pooling`` to factorize Wormhole Pooling vs Random Walk cleanly!
    """

    def __init__(
        self,
        forest: PhilosophicalForest,
        *,
        mode: str = "bm25",  # "bm25" | "idf_cosine"
        wormhole_equivalence_pooling: bool = False,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if mode not in {"bm25", "idf_cosine"}:
            raise ValueError(f"Unsupported FlatEqualMetadataIndex mode: {mode}")
        self._forest_ref = forest
        self.mode = mode
        self.wormhole_equivalence_pooling = bool(wormhole_equivalence_pooling)
        self.k1 = float(k1)
        self.b = float(b)
        self.refresh_count: int = 0
        self._indexed_forest_state: tuple[Any, ...] = ()
        self._build_from_forest(forest)

    def _build_from_forest(self, forest: PhilosophicalForest) -> None:
        t0 = time.perf_counter()
        adj_wh, _ = _extract_node_incident_metadata(forest)
        self.records: list[dict[str, Any]] = []
        active_nodes_map: dict[str, dict[str, Any]] = {}
        df: dict[str, int] = {}

        for nid, node in sorted(forest.nodes.items()):
            if node["utility_ledger"].get("archived_low_utility"):
                continue
            if node["claim_contract"].get("evidence_status") != "SUPPORTED_IN_SCOPE":
                continue
            active_nodes_map[nid] = deepcopy(node)
            sigs = validate_observable_query_signatures(list(node.get("probe_signatures", ())))
            for s in set(sigs):
                df[s] = df.get(s, 0) + 1

            inc_edges = adj_wh.get(nid, [])
            rep_wh = deepcopy(inc_edges[0]) if inc_edges else None
            wh_chain: list[dict[str, Any]] = []
            if rep_wh is not None:
                p_wid = rep_wh.get("role_and_observation_map", {}).get("chain_parent_wormhole")
                if p_wid and p_wid in forest.horizontal_wormhole_edges:
                    wh_chain = [deepcopy(forest.horizontal_wormhole_edges[p_wid]), deepcopy(rep_wh)]
                else:
                    wh_chain = [deepcopy(rep_wh)]

            self.records.append({
                "node_id": nid,
                "target_node": deepcopy(node),
                "sigs": sigs,
                "wormhole_edge": rep_wh,
                "wormhole_path_edges": wh_chain,
                "tie_key": _node_content_tie_key(node),
            })

        if self.wormhole_equivalence_pooling:
            self.wh_component_of = _build_same_schema_wormhole_components(
                sorted(active_nodes_map.keys()), active_nodes_map, adj_wh
            )
        else:
            self.wh_component_of = {r["node_id"]: [r["node_id"]] for r in self.records}

        n_docs = max(1, len(self.records))
        self.idf: dict[str, float] = {
            s: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for s, freq in df.items()
        }
        self.avg_dl = (sum(len(r["sigs"]) for r in self.records) / n_docs) if self.records else 1.0
        self.build_wall_seconds = time.perf_counter() - t0
        self.index_build_compute_cost = round(0.20 * len(self.records), 4)
        self._indexed_forest_state = forest.forest_state_signature()

    def ensure_fresh(self) -> float:
        cur_state = self._forest_ref.forest_state_signature()
        if cur_state == self._indexed_forest_state:
            return 0.0
        self._build_from_forest(self._forest_ref)
        self.refresh_count += 1
        return float(self.index_build_compute_cost)

    def retrieve(
        self,
        observed_signatures: list[str],
        *,
        task_view: dict[str, Any],
        top_k: int = 5,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        refresh_cost = self.ensure_fresh()
        q_sigs = validate_observable_query_signatures(observed_signatures)
        q_set = set(q_sigs)
        req_arity = task_view.get("active_arity")
        goal = task_view.get("goal_kind")

        q_norm = math.sqrt(sum(self.idf.get(s, 0.5) ** 2 for s in q_set)) or 1.0
        raw_doc_scores: dict[str, float] = {}
        overlaps_by_nid: dict[str, list[str]] = {}
        for rec in self.records:
            doc_sigs = rec["sigs"]
            overlap = sorted(q_set & set(doc_sigs))
            overlaps_by_nid[rec["node_id"]] = overlap
            if not overlap:
                raw_doc_scores[rec["node_id"]] = 0.0
                continue
            if self.mode == "bm25":
                dl = max(1, len(doc_sigs))
                denom_norm = 1.0 - self.b + self.b * (dl / max(1e-6, self.avg_dl))
                sc = sum(
                    self.idf.get(s, 0.5) * ((self.k1 + 1.0) / (1.0 + self.k1 * denom_norm))
                    for s in overlap
                )
            else:
                d_norm = math.sqrt(sum(self.idf.get(s, 0.5) ** 2 for s in set(doc_sigs))) or 1.0
                sc = sum(self.idf.get(s, 0.5) ** 2 for s in overlap) / (q_norm * d_norm)
            raw_doc_scores[rec["node_id"]] = float(sc)

        scored: list[dict[str, Any]] = []
        for rec in self.records:
            nid = rec["node_id"]
            node = rec["target_node"]
            node_arity = node["observation_map"].get("parameter_semantics", {}).get("active_arity")
            if req_arity is not None and node_arity is not None and int(node_arity) != int(req_arity):
                continue
            rep_wh = rec["wormhole_edge"]
            if rep_wh is not None and goal and rep_wh.get("compatible_goal_kinds"):
                if goal not in rep_wh["compatible_goal_kinds"]:
                    continue

            comp_members = self.wh_component_of.get(nid, [nid])
            pooled_sc = sum(raw_doc_scores.get(m, 0.0) for m in comp_members)
            if pooled_sc <= 1e-12:
                continue

            scored.append({
                "node_id": nid,
                "target_node": deepcopy(node),
                "doc_score": round(raw_doc_scores.get(nid, 0.0), 8),
                "score": round(float(pooled_sc), 8),
                "matched_signatures": overlaps_by_nid.get(nid, []),
                "reached_only_by_graph_walk": False,
                "wormhole_edge": deepcopy(rep_wh),
                "wormhole_path_edges": deepcopy(rec["wormhole_path_edges"]),
                "entry_kind": ("horizontal_mid_wormhole" if rec["wormhole_path_edges"] else "local_tree_node"),
                "granted_tier": "retrieve_candidate",
                "_tie_key": rec["tie_key"],
            })

        scored.sort(key=lambda item: (-item["score"], -item["doc_score"], item["_tie_key"]))
        deduped: list[dict[str, Any]] = []
        seen_schemas: set[str] = set()
        for item in scored:
            st = str(item["target_node"]["claim_contract"]["conclusion_or_action"].get("schema_tag") or item["_tie_key"])
            if st in seen_schemas:
                continue
            seen_schemas.add(st)
            item_clean = {k: v for k, v in item.items() if k != "_tie_key"}
            deduped.append(item_clean)
            if len(deduped) >= top_k:
                break

        q_compute = round(1.0 + 0.08 * len(self.records) + refresh_cost, 4)
        return {
            "candidates": deduped,
            "non_direct_walk_recalled_count": 0,
            "query_compute_cost": q_compute,
            "index_refreshed": refresh_cost > 0.0,
            "index_refresh_compute_cost": refresh_cost,
            "query_wall_seconds": time.perf_counter() - t0,
        }
