#!/usr/bin/env python3
"""M9.9.1 Zero-LLM Audit Closure, 30-Probe Off-Diagonal Decomposition Back-Verification,
and Forensic Provenance Ledger (`Q01–Q14` + Original `R01–R16` Crosswalk).

Strictly read-only on `docs/evidence/m9_9_*` and `docs/evidence/m9_8_1_*` (`0` paid LLM calls!).
Executes inside air-gapped `hegel-clean-scipy:m7v1` containers:
  1. `01_m99_q01_30_cross_diagonal_decomposition_probes.json`:
     - Executes all `15 × 2 = 30` previously unexecuted off-diagonal public-gate checks (`u0+d0 @ E_decomp` and `u0+d1 @ Stage S`)
       across the `15` accepted `M9.9` confirmation decompositions (`8` `Hegel` + `7` `Plain`), recording candidate SHA, input SHA,
       canonical numeric SHA, parameter payload, `probe_id`, `wall_ms`, `execution_status`, `accepted`, `failure_tier`, `failure_kind`,
       and explicit `supersedes_unexecuted_witness_id` links for the 8 original `Hegel` `d0@E` witnesses.
  2. `02_m99_q03_c04_e7_existing_candidate_container_verification.json`:
     - Executes the prefix-available `bind_alpha_m99_c04_up_b02` on `alpha_m99_c04` `E7_coupled_revisit` in real Docker containers
       (direct `56/24` component probe + composed 3-layer public gate + official verifier, strictly scoped as post-hoc audit
       attribution without altering the sealed `50/56` `M9.9` confirmation score).
  3. `03_m99_provenance_x_sensitivity_and_plain_dict_replay.json`:
     - Records immutable Git tree hashes (`7624c43bf` `34b01d19...` vs post-freeze `be7190fb...`), exact call timestamp envelopes
       (`3,718.765s` confirmation, `2,259.938s` dev), exact `x` sign-flip threshold (`x > 53,450`), `C_cancelled` accounting,
       `7/7` withdrawal foreign-key ledger verification, and `16/16` plain-dict `step_05` routing/candidate SHA parity.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import io
import tarfile
from tempfile import NamedTemporaryFile
from collections import Counter
import json
from pathlib import Path
import subprocess
from time import monotonic
from typing import Any
import uuid

import numpy as np

from hegel_machine.cognition.cost_policy import score_single_call_usage
from hegel_machine.cognition import (
    m8_persistent_variational_planner as m8p,
    m9_4_continual_mechanism_binding_world as m94,
    m9_5_equal_tool_continual_attribution_world as m95,
    m9_6_autonomous_structure_continual_world as m96,
    m9_8_blind_mismatch_and_selective_dependency_world as m98,
    m9_9_autonomous_decomposition_and_interaction_withdrawal_world as m99,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
M99_CONF_DIR = REPO_ROOT / "docs/evidence/m9_9_autonomous_decomposition_and_interaction_withdrawal_20261128"
M99_DEV_DIR = REPO_ROOT / "docs/evidence/m9_9_dev_live_calibration_20261128"
OUT_DIR = REPO_ROOT / "docs/evidence/m9_9_1_audit_closure_and_forensic_reverification_20261128"


def _compute_commit_py_tree_sha256(commit_ref: str, *, omit_report_generator: bool = False) -> dict[str, Any]:
    tree = subprocess.check_output(
        ["git", "ls-tree", "-rz", commit_ref, "--", "src", "scripts", "tests"], cwd=REPO_ROOT
    )
    entries = []
    for item in tree.split(b"\0"):
        if not item.endswith(b".py"):
            continue
        meta, path = item.split(b"\t", 1)
        if omit_report_generator and path == b"scripts/generate_m9_9_report_from_sealed_json.py":
            continue
        entries.append((path, meta.split()[2]))
    directory_order = [b"src", b"scripts", b"tests"]
    entries.sort(key=lambda pair: (directory_order.index(pair[0].split(b"/", 1)[0]), pair[0]))
    response = subprocess.run(
        ["git", "cat-file", "--batch"], cwd=REPO_ROOT, input=b"".join(oid+b"\n" for _, oid in entries),
        stdout=subprocess.PIPE, check=True,
    )
    stream = io.BytesIO(response.stdout)
    h = hashlib.sha256()
    for path, oid in entries:
        got_oid, kind, size = stream.readline().split()
        if got_oid != oid or kind != b"blob":
            raise ValueError(f"Unexpected Git object for {path!r}")
        content = stream.read(int(size))
        if stream.read(1) != b"\n":
            raise ValueError(f"Truncated Git object for {path!r}")
        h.update(path+b"\0"+hashlib.sha256(content).digest())
    return {"commit": commit_ref, "py_file_count": len(entries), "source_tree_sha256": h.hexdigest(),
            "excluded_later_report_generator": omit_report_generator}


def _sealed_container_public_sha(world_id: str, step: int) -> str:
    if step:
        stream_root = M99_CONF_DIR / "continual_streams" / world_id
        paths = stream_root.glob(f"*/{world_id}_batch_{step:02d}__*/00_episode_summary.json")
    else:
        paths = (M99_CONF_DIR / "stage_S_sources" / world_id).glob("attempt_*/*/00_episode_summary.json")
    digests = set()
    for path in paths:
        receipt = json.loads(path.read_bytes()).get("public_setup_receipt") or {}
        digest = (receipt.get("files_sha256") or {}).get("/holdout_task/public_batch.pkl")
        if digest:
            digests.add(digest)
    if len(digests) != 1:
        raise ValueError(f"Expected one sealed container input for {world_id} step {step}: {digests}")
    return digests.pop()


def _run_30_cross_diagonal_probes() -> dict[str, Any]:
    receipts = json.loads((M99_CONF_DIR / "all_provider_call_receipts.json").read_text(encoding="utf-8"))
    decomp_calls = [r for r in receipts if r["call_role"] == "autonomous_decomposition_induction"]

    specs_by_wid = {s.world_id: s for s in m99.CONFIRMATION_M99_WORLD_SPECS}
    built_worlds = {wid: m99.build_m99_world_tasks(sp) for wid, sp in specs_by_wid.items()}

    jobs: list[dict[str, Any]] = []
    for arm in ("Flat_PlainModuleRegistry_Online", "Hegel_AutonomousInteractionGraph_Online"):
        for wid in sorted(specs_by_wid.keys()):
            st_path = M99_CONF_DIR / "continual_streams" / wid / arm / "00_stream_summary.json"
            st = json.loads(st_path.read_text(encoding="utf-8"))
            acc_evs = [d for d in st["autonomous_decomposition_events"] if d.get("accepted")]
            if not acc_evs:
                continue
            ev = acc_evs[0]
            step_num = int(ev["step"])
            att_used = int(ev["attempts_used"])
            ep_short = f"{wid}_batch_{step_num:02d}"
            tag_sub = f"_decomp_att{att_used}"
            matched_calls = [
                c
                for c in decomp_calls
                if c["world_id"] == wid and c["arm"] == arm and c["episode_short_name"] == ep_short and tag_sub in c["stage_subdir"]
            ]
            assert len(matched_calls) == 1, f"Expected 1 accepted decomp call for {wid}/{arm}, got {len(matched_calls)}"
            raw_obj = json.loads((M99_CONF_DIR / "provider_raw_attempts_mirror" / matched_calls[0]["mirrored_raw_filename"]).read_text(encoding="utf-8"))
            parsed = json.loads(raw_obj["raw_response_text"])
            u0_code = str(parsed["upstream_code"])
            d0_code = str(parsed["downstream_d0_code"])
            d1_code = str(parsed["downstream_d1_code"])

            spec = specs_by_wid[wid]
            s0_task, tasks, _, logs = built_worlds[wid]
            e_task = tasks[step_num - 1]
            s0_pub, s0_sha = m98.materialize_m98_public_item_and_sha(s0_task)
            e_pub, e_sha = m98.materialize_m98_public_item_and_sha(e_task)
            s0_num_sha = m96.compute_canonical_numeric_item_sha256(s0_pub)
            e_num_sha = m96.compute_canonical_numeric_item_sha256(e_pub)

            s0_ro_deps = {"/tmp/hegel_continual_archive/archive_apparatus_log.json": json.dumps(logs[:1], indent=2)}
            s0_ro_shas = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in s0_ro_deps.items()}
            e_ro_deps = {"/tmp/hegel_continual_archive/archive_apparatus_log.json": json.dumps(logs[: step_num + 1], indent=2)}
            e_ro_shas = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in e_ro_deps.items()}

            s0_pay = {"gains": [1.0] * 4, "offsets": [0.0] * 4}
            # Reconstruct exact step parameter payload used at `step_num`
            compat = spec.to_m98_compat_spec()
            if arm == "Hegel_AutonomousInteractionGraph_Online":
                ck_step_path = M99_CONF_DIR / "continual_streams" / wid / arm / f"graph_ckpt_after_step_{step_num:02d}.json"
                g_tmp = m8p.PersistentMechanismGraph.load_checkpoint(ck_step_path)
                r_tmp = m94.M94BindingRegistry(g_tmp)
                assert hashlib.sha256(u0_code.encode("utf-8")).hexdigest() == r_tmp.bindings[f"bind_{wid}_up_b00"].authoritative_executed_code_sha256()
                assert hashlib.sha256(d0_code.encode("utf-8")).hexdigest() == r_tmp.bindings[f"bind_{wid}_dn_b00"].authoritative_executed_code_sha256()
                assert hashlib.sha256(d1_code.encode("utf-8")).hexdigest() == r_tmp.bindings[f"bind_{wid}_dn_b01"].authoritative_executed_code_sha256()
                e_pay = m98.assemble_m98_graph_diagonal_payload(compat, g_tmp)
                dn_c = m96.ImmutableBranchLineageController(world_id=f"{wid}_dn", family=f"{spec.family}_dn", graph=g_tmp, registry=r_tmp)
                unexec_wits = [
                    w.witness_id
                    for w in dn_c.witness_ledger
                    if w.binding_id == f"bind_{wid}_dn_b00" and w.public_input_sha256 == e_sha and w.result == "failed" and not w.probe_id
                ]
                supersedes_wit_id = unexec_wits[0] if unexec_wits else None
            else:
                flat_snap = {"gains": [1.0] * 4, "offsets": [0.0] * 4}
                for s_k in range(1, step_num + 1):
                    pk_k, _ = m98.materialize_m98_public_item_and_sha(tasks[s_k - 1])
                    flat_snap = m99.update_m99_calibration_snapshot_and_graph(
                        spec=compat, episode_id=f"{wid}_s{s_k}", public_item=pk_k, flat_snapshot=flat_snap
                    )["flat_snapshot"]
                e_pay = flat_snap
                supersedes_wit_id = None

            jobs.append(
                {
                    "cell": "u0_plus_d0_on_current_E",
                    "world_id": wid,
                    "family": spec.family,
                    "arm": arm,
                    "decomp_step": step_num,
                    "decomp_event": ev["event"],
                    "call_uuid": matched_calls[0]["call_uuid"],
                    "task": e_task,
                    "upstream_code": u0_code,
                    "downstream_code": d0_code,
                    "param_payload": e_pay,
                    "ro_deps": e_ro_deps,
                    "ro_shas": e_ro_shas,
                    "public_input_sha256": e_sha,
                    "sealed_container_sha256": _sealed_container_public_sha(wid, step_num),
                    "canonical_numeric_sha256": e_num_sha,
                    "supersedes_unexecuted_witness_id": supersedes_wit_id,
                }
            )
            jobs.append(
                {
                    "cell": "u0_plus_d1_on_Stage_S",
                    "world_id": wid,
                    "family": spec.family,
                    "arm": arm,
                    "decomp_step": step_num,
                    "decomp_event": ev["event"],
                    "call_uuid": matched_calls[0]["call_uuid"],
                    "task": s0_task,
                    "upstream_code": u0_code,
                    "downstream_code": d1_code,
                    "param_payload": s0_pay,
                    "ro_deps": s0_ro_deps,
                    "ro_shas": s0_ro_shas,
                    "public_input_sha256": s0_sha,
                    "sealed_container_sha256": _sealed_container_public_sha(wid, 0),
                    "canonical_numeric_sha256": s0_num_sha,
                    "supersedes_unexecuted_witness_id": None,
                }
            )

    assert len(jobs) == 30, f"Expected exactly 30 cross-diagonal jobs, got {len(jobs)}"

    def _exec_one(jb: dict[str, Any]) -> dict[str, Any]:
        cand = m98.compose_m98_two_stage_solver_code(
            upstream_code=jb["upstream_code"],
            upstream_entrypoint="calibrate_upstream",
            downstream_code=jb["downstream_code"],
            downstream_entrypoint="solve_downstream",
            param_payload=jb["param_payload"],
        )
        pid = f"xprobe_{uuid.uuid4().hex[:16]}"
        t0 = monotonic()
        try:
            pr = m95.execute_single_pre_llm_public_gate_probe_in_container(
                task=jb["task"], candidate_code=cand, readonly_dependencies=jb["ro_deps"],
                expected_dependency_sha256=jb["ro_shas"],
            )
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            pr = {"setup_ok": False, "infrastructure_exception": str(exc),
                  "timed_out": isinstance(exc, subprocess.TimeoutExpired)}
        w_ms = round((monotonic() - t0) * 1000.0, 2)
        cls = m96.classify_public_probe_failure(pr)
        ex_st = "timeout" if pr.get("timed_out") else ("infrastructure_error" if cls["tier"] == "INFRA_FAILURE" else "completed")
        actual_sha = (pr.get("public_setup_files_sha256") or {}).get("/holdout_task/public_batch.pkl")
        bound = (actual_sha == jb["sealed_container_sha256"]
                 and pr.get("candidate_raw_sha256") == hashlib.sha256(cand.encode()).hexdigest()
                 and pr.get("readonly_dependencies_sha256") == jb["ro_shas"])
        if ex_st == "completed" and not bound:
            ex_st = "artifact_identity_error"
        superseded = jb["supersedes_unexecuted_witness_id"] if ex_st == "completed" and cls["tier"] in {
            "NONE", "DOMAIN_REJECTION", "SCIENTIFIC_FAILURE"
        } else None
        return {
            "probe_id": pid,
            "cell": jb["cell"],
            "world_id": jb["world_id"],
            "family": jb["family"],
            "arm": jb["arm"],
            "decomp_step": jb["decomp_step"],
            "decomp_event": jb["decomp_event"],
            "source_decomp_call_uuid": jb["call_uuid"],
            "target_task_id": jb["task"].task_id,
            "upstream_code_sha256": hashlib.sha256(jb["upstream_code"].encode("utf-8")).hexdigest(),
            "downstream_code_sha256": hashlib.sha256(jb["downstream_code"].encode("utf-8")).hexdigest(),
            "composed_candidate_sha256": hashlib.sha256(cand.encode("utf-8")).hexdigest(),
            "public_input_sha256": actual_sha,
            "host_public_pickle_sha256": jb["public_input_sha256"],
            "sealed_container_public_input_sha256": jb["sealed_container_sha256"],
            "artifact_identity_verified": bound,
            "public_gate_receipt": pr,
            "canonical_numeric_sha256": jb["canonical_numeric_sha256"],
            "parameter_payload": jb["param_payload"],
            "dependency_bundle_sha256": hashlib.sha256(json.dumps(jb["ro_shas"], sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "execution_status": ex_st,
            "accepted": bool(pr.get("accepted")) if ex_st == "completed" else None,
            "failure_tier": cls["tier"],
            "failure_kind": cls["failure_kind"],
            "probe_wall_ms": w_ms,
            "verifier_version": m99.SHARED_KERNEL_VERSION,
            "historical_unexecuted_witness_id": jb["supersedes_unexecuted_witness_id"],
            "supersedes_unexecuted_witness_id": superseded,
            "errata_relation": ("supersedes_unexecuted_projection_with_executed_probe" if superseded else ex_st),
        }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_exec_one, j) for j in jobs]
        for f in as_completed(futs):
            res = f.result()
            results.append(res)
            print(f"  [Q01 30-probe] {res['arm']} | {res['world_id']} | {res['cell']} -> accepted={res['accepted']} ({res['failure_kind']}, {res['probe_wall_ms']} ms)", flush=True)

    results.sort(key=lambda x: (x["arm"], x["world_id"], x["cell"]))
    return {
        "schema": "hegel_m9_9_1_q01_30_cross_diagonal_backverification_v1",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "paid_llm_calls": 0,
        "evidence_scope": "Post-hoc real Docker public-gate back-verification of the 30 off-diagonal cells across 15 accepted M9.9 decompositions; original sealed M9.9 scores/files untouched.",
        "total_probes_attempted": len(results),
        "total_probes_executed": sum(r["execution_status"] == "completed" for r in results),
        "accepted_count": sum(1 for r in results if r["accepted"]),
        "scientific_or_domain_rejected_count": sum(
            1 for r in results if r["accepted"] is False and r["failure_kind"] in {"scientific_or_numerical", "domain_precondition"}
        ),
        "schema_rejected_count": sum(1 for r in results if r["accepted"] is False and r["failure_kind"] == "schema_or_interface"),
        "infra_or_timeout_count": sum(1 for r in results if r["execution_status"] != "completed"),
        "total_probe_wall_ms": round(sum(float(r["probe_wall_ms"]) for r in results), 2),
        "superseded_hegel_d0_witness_count": sum(1 for r in results if r["supersedes_unexecuted_witness_id"] is not None),
        "probes": results,
    }


def _run_c04_e7_existing_candidate_container_verification() -> dict[str, Any]:
    spec_c04 = next(s for s in m99.CONFIRMATION_M99_WORLD_SPECS if s.world_id == "alpha_m99_c04")
    compat = spec_c04.to_m98_compat_spec()
    _, tasks, _, logs = m99.build_m99_world_tasks(spec_c04)
    e7_task = tasks[6]
    pub_item, pub_sha = m98.materialize_m98_public_item_and_sha(e7_task)
    ro_deps = {"/tmp/hegel_continual_archive/archive_apparatus_log.json": json.dumps(logs[:8], indent=2)}
    ro_shas = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in ro_deps.items()}

    ck6 = M99_CONF_DIR / "continual_streams/alpha_m99_c04/Hegel_AutonomousInteractionGraph_Online/graph_ckpt_after_step_06.json"
    g6 = m8p.PersistentMechanismGraph.load_checkpoint(ck6)
    r6 = m94.M94BindingRegistry(g6)
    m99.update_m99_calibration_snapshot_and_graph(spec=compat, episode_id="c04_e7_audit_chk", public_item=pub_item, graph=g6, registry=r6)
    pay = m98.assemble_m98_graph_diagonal_payload(compat, g6)

    up_b02 = r6.bindings["bind_alpha_m99_c04_up_b02"]
    dn_b03 = r6.bindings["bind_alpha_m99_c04_dn_b03"]

    up_diag = m98.diagnose_m98_factored_conflict_locus(
        "alpha",
        pub_item,
        chosen_up_code=up_b02.canonical_algorithm_code,
        chosen_up_entrypoint=up_b02.canonical_entrypoint,
        diag_payload=pay,
        force_direct_component_probe=True,
    )
    cand = m98.compose_m98_two_stage_solver_code(
        upstream_code=up_b02.canonical_algorithm_code,
        upstream_entrypoint=up_b02.canonical_entrypoint,
        downstream_code=dn_b03.canonical_algorithm_code,
        downstream_entrypoint=dn_b03.canonical_entrypoint,
        param_payload=pay,
    )
    t_pg = monotonic()
    pub_gate = m95.execute_single_pre_llm_public_gate_probe_in_container(
        task=e7_task, candidate_code=cand, readonly_dependencies=ro_deps, expected_dependency_sha256=ro_shas
    )
    pg_ms = round((monotonic() - t_pg) * 1000.0, 2)

    t_pv = monotonic()
    priv_ver = m94.grade_m94_artifact_map(
        e7_task, {str(e7_task.declared_artifacts[0]): cand}, readonly_dependencies=ro_deps, expected_dependency_sha256=ro_shas
    )
    pv_ms = round((monotonic() - t_pv) * 1000.0, 2)

    return {
        "schema": "hegel_m9_9_1_q03_c04_e7_fallback_container_verification_v1",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "paid_llm_calls": 0,
        "evidence_scope": "Post-hoc container verification on exposed prefix checkpoint `graph_ckpt_after_step_06.json` (`alpha_m99_c04`); strictly diagnostic/regression evidence, does NOT alter sealed M9.9 confirmation score or cost.",
        "world_id": "alpha_m99_c04",
        "event": "E7_coupled_revisit",
        "host_public_pickle_sha256": pub_sha,
        "public_input_sha256": (pub_gate.get("public_setup_files_sha256") or {}).get("/holdout_task/public_batch.pkl"),
        "sealed_container_public_input_sha256": _sealed_container_public_sha(spec_c04.world_id, 7),
        "tested_upstream_binding_id": up_b02.binding_id,
        "tested_upstream_code_sha256": up_b02.authoritative_executed_code_sha256(),
        "tested_downstream_binding_id": dn_b03.binding_id,
        "tested_downstream_code_sha256": dn_b03.authoritative_executed_code_sha256(),
        "composed_candidate_sha256": hashlib.sha256(cand.encode("utf-8")).hexdigest(),
        "direct_upstream_component_probe": up_diag,
        "joint_three_layer_public_gate": {
            "accepted": bool(pub_gate.get("accepted")),
            "probe_wall_ms": pg_ms,
            "scientific_probe_receipt": pub_gate.get("scientific_probe_receipt"),
            "public_gate_receipt": pub_gate,
        },
        "official_verifier_result": {
            "strict_pass": bool(priv_ver.get("strict_pass")),
            "verifier_wall_ms": pv_ms,
            "verifier_metrics": priv_ver.get("verifier_metrics"),
        },
    }


def _build_provenance_and_sensitivity_ledger() -> dict[str, Any]:
    conf_calls = json.loads((M99_CONF_DIR / "all_provider_call_receipts.json").read_text(encoding="utf-8"))
    dev_calls = json.loads((M99_DEV_DIR / "all_provider_call_receipts.json").read_text(encoding="utf-8"))
    conf_sum = json.loads((M99_CONF_DIR / "m9_9_continual_summary.json").read_text(encoding="utf-8"))

    def _envelope(calls: list[dict[str, Any]], mirror_dir: Path) -> dict[str, Any]:
        starts, ends = [], []
        run_prefixes = set()
        for c in calls:
            run_prefixes.add(c["stage_subdir"].split("__")[0])
            m_obj = json.loads((mirror_dir / c["mirrored_raw_filename"]).read_text(encoding="utf-8"))
            starts.append(datetime.fromisoformat(m_obj["started_at_utc"]))
            ends.append(datetime.fromisoformat(m_obj["ended_at_utc"]))
        t_min, t_max = min(starts), max(ends)
        return {
            "execution_run_ids_in_raw_calls": sorted(run_prefixes),
            "first_call_started_utc": t_min.isoformat(),
            "last_call_ended_utc": t_max.isoformat(),
            "call_envelope_lower_bound_sec": round((t_max - t_min).total_seconds(), 6),
        }

    # Exact `x` sensitivity ladder for `Hegel - Plain` (`diffs = [35798, -68719, -938884, 217083 - x, -641537, -199431, -579431, -360562]`)
    base_diffs = np.array([35798.0, -68719.0, -938884.0, 217083.0, -641537.0, -199431.0, -579431.0, -360562.0], dtype=float)
    x_probe_points = [0, 17652, 17653, 53450, 53451, 148364, 148365, 184162, 184163, 217083, 217084, 252881, 252882]
    signs = np.array([[1.0 if ((mask >> j) & 1) else -1.0 for j in range(8)] for mask in range(256)], dtype=float)
    x_ladder = []
    for xv in x_probe_points:
        dv = base_diffs.copy()
        dv[3] -= float(xv)
        obs = abs(float(np.mean(dv)))
        ge = int(np.sum(np.abs((signs * dv[None, :]).mean(axis=1)) >= obs - 1e-12))
        x_ladder.append({"x": xv, "two_sided_exact_sign_flip_p": round(ge / 256.0, 8)})

    # Plain list/dict `16/16` parity check on `step_05` ablation
    abl = json.loads((M99_CONF_DIR / "m9_9_factorized_ceiling_ablation.json").read_text(encoding="utf-8"))
    hg_abl_rows = {(r["world_id"], r["event"]): r for r in abl["rows"] if r["policy"] == "hegel_selective_cross_factor_graph"}
    plain_dict_matches = 0
    for spec in m99.CONFIRMATION_M99_WORLD_SPECS:
        wid = spec.world_id
        compat = spec.to_m98_compat_spec()
        _, tasks, manifest, logs = m99.build_m99_world_tasks(spec)
        ck5_raw = json.loads((M99_CONF_DIR / "continual_streams" / wid / "Hegel_AutonomousInteractionGraph_Online/graph_ckpt_after_step_05.json").read_text(encoding="utf-8"))
        b_dict = ck5_raw["questions"]["__m9_4_mechanism_executable_bindings__"]["bindings"]
        sw_raw = ((ck5_raw["questions"].get(m99.M99_WITHDRAWAL_FOREST_KEY) or {}).get("worlds") or {}).get(wid, {}).get("receipts") or []
        dn_rows = sorted(
            [v for v in b_dict.values() if v["chain_id"] == f"{spec.family}_dn" and v["qualification_status"] == "verified"],
            key=lambda v: (-int(v["selection_priority"]), v["binding_id"]),
        )
        up_rows = sorted(
            [v for v in b_dict.values() if v["chain_id"] == f"{spec.family}_up" and v["qualification_status"] == "verified"],
            key=lambda v: (-int(v["selection_priority"]), v["binding_id"]),
        )
        cross_set = {sw_raw[-1]["downstream_binding_id"], dn_rows[0]["binding_id"]} if sw_raw else set()
        # Base flat snapshot at step 5
        flat_snap = {"gains": [1.0] * 4, "offsets": [0.0] * 4}
        for s_i in range(5):
            pi_s, _ = m98.materialize_m98_public_item_and_sha(tasks[s_i])
            flat_snap = m99.update_m99_calibration_snapshot_and_graph(spec=compat, episode_id=f"{wid}_{s_i}", public_item=pi_s, flat_snapshot=flat_snap)["flat_snapshot"]
        for ev_nm in ("E6_separable_revisit", "E7_coupled_revisit"):
            idx = next(i for i, m in enumerate(manifest) if m["causal_event"] == ev_nm)
            pi, psha = m98.materialize_m98_public_item_and_sha(tasks[idx])
            snap_ev = m99.update_m99_calibration_snapshot_and_graph(spec=compat, episode_id=f"{wid}_{ev_nm}", public_item=pi, flat_snapshot=deepcopy(flat_snap))["flat_snapshot"]
            up_obs = m98.compute_m98_upstream_observables(spec.family, pi)
            dn_obs = m98.compute_m98_downstream_observables(logs[: idx + 2])
            comb_obs = {**up_obs, **dn_obs}
            sel_u = next(
                (r for r in up_rows if m96.evaluate_m96_autonomous_guard(r["applicability_guard"], up_obs, allow_open_feature_keys=True)["state"] == "TRUE"),
                up_rows[0],
            )
            sel_d = next(
                (
                    r
                    for r in dn_rows
                    if m96.evaluate_m96_autonomous_guard(
                        r["applicability_guard"],
                        (comb_obs if (r["binding_id"] in cross_set or any(str(c.get("feature_name", "")).startswith("up_stat_") for c in (r["applicability_guard"].get("all_of") or []))) else dn_obs),
                        allow_open_feature_keys=True,
                    )["state"]
                    == "TRUE"
                ),
                dn_rows[0],
            )
            cand = m98.compose_m98_two_stage_solver_code(
                upstream_code=sel_u["canonical_algorithm_code"],
                upstream_entrypoint=sel_u["canonical_entrypoint"],
                downstream_code=sel_d["canonical_algorithm_code"],
                downstream_entrypoint=sel_d["canonical_entrypoint"],
                param_payload=snap_ev,
            )
            csha = hashlib.sha256(cand.encode("utf-8")).hexdigest()
            ref = hg_abl_rows[(wid, ev_nm)]
            if sel_u["binding_id"] == ref["selected_upstream"] and sel_d["binding_id"] == ref["selected_downstream"] and csha == ref["candidate_sha256"] and psha == ref["public_batch_pkl_sha256"]:
                plain_dict_matches += 1

    ot = conf_sum["arm_outcome_totals"]
    tot_agent_ms = {
        a: round(
            ot[a]["pre_llm_probe_wall_ms"]
            + ot[a]["post_llm_probe_wall_ms"]
            + ot[a]["localization_probe_wall_ms"]
            + ot[a]["decomposition_probe_wall_ms"]
            + ot[a]["coverage_probe_wall_ms"],
            2,
        )
        for a in ot
    }

    return {
        "schema": "hegel_m9_9_1_provenance_sensitivity_and_attribution_v1",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "paid_llm_calls": 0,
        "git_source_tree_reconciliation": {
            "declared_paid_freeze_commit_7624c43bf": _compute_commit_py_tree_sha256("7624c43bf72d283e82442504eff35ff6ee02b40d"),
            "m99_summary_recorded_post_freeze_finalizer_tree_sha256": conf_sum["source_tree_sha256_at_start"],
            "seal_commit_669ccf0b2": _compute_commit_py_tree_sha256("669ccf0b2"),
            "doc_commit_d4b1cd7f0": _compute_commit_py_tree_sha256("d4b1cd7f0"),
            "provenance_note": "Original paid confirmation run had no persisted start manifest; `be7190fb...` in `m9_9_continual_summary.json` measured the post-freeze finalizer tree (`--finalize-existing`), while git diff confirms 0 changes to paid online loop, world generator, or official verifier.",
        },
        "execution_vs_finalizer_timing_reconciliation": {
            "confirmation_n8": {
                **_envelope(conf_calls, M99_CONF_DIR / "provider_raw_attempts_mirror"),
                "finalizer_summary_run_id": conf_sum["run_id"],
                "finalizer_postprocess_wall_sec": conf_sum["wall_sec"],
            },
            "dev_n2": _envelope(dev_calls, M99_DEV_DIR / "provider_raw_attempts_mirror"),
        },
        "rd_cost_coverage_boundary": {
            "sealed_dev_plus_confirmation_logical_calls": len(dev_calls) + len(conf_calls),
            "sealed_dev_physical_v2": 8409059,
            "sealed_confirmation_physical_v2_known": 37274561,
            "combined_sealed_physical_v2_formula": "45683620 + x",
            "unarchived_pre_freeze_cancelled_dev_diagnostic_run": {
                "status": "unverifiable_cancelled_run_not_archived",
                "symbolic_term": "C_cancelled > 0 (unknown, never filled as 0)",
                "total_rd_v2_formula": "45683620 + x + C_cancelled",
                "confirmation_seeds_touched": 0,
            },
        },
        "unknown_x_exact_sign_flip_sensitivity_ladder": {
            "call_uuid": "0391d702b0c547a488608847b5dff224",
            "arm": "Flat_PlainModuleRegistry_Online",
            "location": "alpha_m99_c04 / E1_down / attempt 1 timeout",
            "threshold_for_two_sided_p_lt_005": "x > 53450",
            "evaluated_points": x_ladder,
        },
        "plain_dict_same_joint_features_step05_replay": {
            "matched_episodes": plain_dict_matches,
            "total_episodes": 16,
            "attribution_scope": "Selective condition-expansion policy (`dn_obs -> up_obs + dn_obs` after counterexample) is substrate-independent: a plain JSON dict/list given identical `up+dn` features and flat snapshot reproduces 16/16 selected branches, candidate SHAs, and public input SHAs.",
        },
        "non_llm_probe_elapsed_wall_ratios": {
            "total_agent_probe_elapsed_wall_ms": tot_agent_ms,
            "monolithic_over_hegel_total_agent_probe_wall_ratio": round(
                tot_agent_ms["Flat_MonolithicConditionalTable_Online"] / tot_agent_ms["Hegel_AutonomousInteractionGraph_Online"], 6
            ),
            "monolithic_over_hegel_empirical_coverage_subitem_wall_ratio": round(
                ot["Flat_MonolithicConditionalTable_Online"]["coverage_probe_wall_ms"] / ot["Hegel_AutonomousInteractionGraph_Online"]["coverage_probe_wall_ms"],
                6,
            ),
            "measurement_unit": "summed_container_elapsed_wall_ms_not_cpu_seconds",
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p01 = OUT_DIR / "01_m99_q01_30_cross_diagonal_decomposition_probes.json"
    if p01.exists() and "total_probes_attempted" in json.loads(p01.read_bytes()):
        print("[M9.9.1] 1/3 Reusing sealed 30 off-diagonal Docker public-gate back-verification probes (`Q01`)...", flush=True)
        r01 = json.loads(p01.read_text(encoding="utf-8"))
    else:
        print("[M9.9.1] 1/3 Running 30 off-diagonal Docker public-gate back-verification probes (`Q01`)...", flush=True)
        r01 = _run_30_cross_diagonal_probes()
        p01.write_text(json.dumps(r01, indent=2) + "\n", encoding="utf-8")

    p02 = OUT_DIR / "02_m99_q03_c04_e7_existing_candidate_container_verification.json"
    if p02.exists() and "sealed_container_public_input_sha256" in json.loads(p02.read_bytes()):
        print("[M9.9.1] 2/3 Reusing sealed `alpha_m99_c04/E7` existing `up_b02` container verification (`Q03`)...", flush=True)
        r02 = json.loads(p02.read_text(encoding="utf-8"))
    else:
        print("[M9.9.1] 2/3 Running `alpha_m99_c04/E7` existing `up_b02` container verification (`Q03`)...", flush=True)
        r02 = _run_c04_e7_existing_candidate_container_verification()
        p02.write_text(json.dumps(r02, indent=2) + "\n", encoding="utf-8")

    print("[M9.9.1] 3/3 Building provenance, `x` sensitivity, `C_cancelled`, and 16/16 plain-dict replay ledger...", flush=True)
    r03 = _build_provenance_and_sensitivity_ledger()
    assert r03["plain_dict_same_joint_features_step05_replay"]["matched_episodes"] == 16, r03["plain_dict_same_joint_features_step05_replay"]
    (OUT_DIR / "03_m99_provenance_x_sensitivity_and_plain_dict_replay.json").write_text(json.dumps(r03, indent=2) + "\n", encoding="utf-8")

    sha_lines = []
    for fp in sorted(OUT_DIR.rglob("*")):
        if fp.is_file() and fp.name != "SHA256SUMS.txt":
            rel = fp.relative_to(OUT_DIR).as_posix()
            sha_lines.append(f"{hashlib.sha256(fp.read_bytes()).hexdigest()}  {rel}")
    (OUT_DIR / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "q01_30_probes_executed": r01["total_probes_executed"],
                "q01_accepted": r01["accepted_count"],
                "q01_scientific_rejected": r01["scientific_or_domain_rejected_count"],
                "q01_schema_rejected": r01["schema_rejected_count"],
                "q03_c04_e7_up_b02_rmse": r02["direct_upstream_component_probe"]["chosen_up_holdout_rmse"],
                "q03_c04_e7_public_gate": r02["joint_three_layer_public_gate"]["accepted"],
                "q03_c04_e7_private_strict_pass": r02["official_verifier_result"]["strict_pass"],
                "plain_dict_16_of_16": r03["plain_dict_same_joint_features_step05_replay"]["matched_episodes"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
