#!/usr/bin/env python3
"""M9.9.1 Zero-LLM Audit Closure, 30-Probe Off-Diagonal Decomposition Back-Verification,
and Forensic Provenance Ledger (`Q01–Q14` + Original `R01–R16` Crosswalk).

Strictly read-only on `docs/evidence/m9_9_*` and `docs/evidence/m9_8_1_*` (`0` paid LLM calls!).
Executes inside air-gapped `hegel_clean_scipy_sandbox:v1` containers:
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
from functools import lru_cache
import argparse
import gzip
import hashlib
import io
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
        ep = json.loads(path.read_bytes())
        setup = (ep.get("public_setup_receipt") or {}).get("files_sha256") or {}
        trial = (ep.get("pre_llm_trial_receipt") or {}).get("probe_result") or {}
        sci = trial.get("scientific_probe_receipt") or {}
        signed_setup = (sci.get("public_setup_files_sha256") or {}) if sci.get("signed_by_host_probe_executor") else {}
        for manifest in (setup, signed_setup):
            digest = manifest.get("/holdout_task/public_batch.pkl")
            if digest:
                digests.add(digest)
    if len(digests) != 1:
        raise ValueError(f"Expected one sealed container input for {world_id} step {step}: {digests}")
    return digests.pop()


@lru_cache(maxsize=1)
def _sandbox_identity() -> dict:
    image = m95.CLEAN_SCIPY_SANDBOX_IMAGE
    digest = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True).strip()
    sealed = json.loads((M99_CONF_DIR / "m9_9_continual_summary.json").read_bytes())["sandbox_image_digest"]
    if digest != sealed:
        raise ValueError(f"Sandbox image differs from the sealed M9.9 image: {digest} != {sealed}")
    return {"image": image, "image_id": digest, "sealed_image_id": sealed}


def _bound_host_measurement(gate: dict, public_sha: str) -> bool:
    sci = gate.get("scientific_probe_receipt") or {}
    return (sci.get("signed_by_host_probe_executor") is True and sci.get("execution_status") == "completed"
            and sci.get("public_input_digest") == public_sha
            and sci.get("candidate_bundle_sha256") == gate.get("candidate_bundle_sha256")
            and sci.get("public_setup_files_sha256") == gate.get("public_setup_files_sha256")
            and bool(sci.get("kernel_version")) and bool(sci.get("measured_scientific_metrics")))


def _run_30_cross_diagonal_probes() -> dict[str, Any]:
    sandbox = _sandbox_identity()
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
        measurement_bound = _bound_host_measurement(pr, actual_sha)
        if ex_st == "completed" and cls["tier"] in {"NONE", "SCIENTIFIC_FAILURE"} and not measurement_bound:
            ex_st = "unverified_scientific_receipt"
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
            "host_measurement_verified": measurement_bound,
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
        "schema": "hegel_m9_9_1_q01_30_cross_diagonal_backverification_v2", "sandbox": sandbox,
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
        "schema": "hegel_m9_9_1_q03_c04_e7_fallback_container_verification_v2", "sandbox": _sandbox_identity(),
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


def _cancelled_campaign_ledger(stage_root: Path | None, sealed_ids: set[str], sealed_known_v2: int) -> dict:
    prefix = "m99_dev_20261127T175948Z__"
    archive = OUT_DIR / "04_m99_cancelled_dev_usage_receipts.jsonl.gz"
    if not archive.is_file():
        files = sorted(stage_root.glob(prefix + "*/attempts_*.json")) if stage_root else []
        if not files:
            raise FileNotFoundError("Provide --cancelled-stage-root to recover the cancelled development receipts")
        # Archive only attribution and billing evidence. Source hashes bind each row to exact provider bytes.
        with NamedTemporaryFile(dir=OUT_DIR, delete=False) as tmp:
            temp_path = Path(tmp.name)
        try:
            with temp_path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
                for source in files:
                    data = source.read_bytes()
                    original = json.loads(data)
                    row = {key: original[key] for key in ("stage_subdir", "call_uuid", "status", "call_seq", "attempt_count")}
                    row["source_receipt_sha256"] = hashlib.sha256(data).hexdigest()
                    row["attempts"] = [
                        {key: attempt[key] for key in ("attempt_index", "usage_known", "prompt_tokens", "cached_tokens",
                                                       "completion_tokens", "response_sha256", "timed_out")}
                        for attempt in original["attempts"]
                    ]
                    if row["status"] != "completed":
                        pending_response = source.parent / f"resp_{row['call_seq']:03d}_att{row['attempt_count']}.json"
                        row["pending_response_bytes"] = pending_response.stat().st_size if pending_response.exists() else None
                    compressed.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
            temp_path.replace(archive)
        finally:
            temp_path.unlink(missing_ok=True)
    ids = set()
    stages = set()
    statuses = Counter()
    unfinished = []
    unknown_attempts = 0
    known_attempts = 0
    known_v2 = 0
    with gzip.open(archive, "rt", encoding="utf-8") as receipts:
        for line in receipts:
            row = json.loads(line)
            stage = row["stage_subdir"]
            cid = row["call_uuid"]
            if not stage.startswith(prefix) or "/" in stage or "\\" in stage or cid in ids or cid in sealed_ids:
                raise ValueError(f"Unbound or repeated cancelled campaign receipt for {cid}")
            if len(row["source_receipt_sha256"]) != 64 or len(row["attempts"]) != row["attempt_count"]:
                raise ValueError(f"Incomplete cancelled campaign projection for {cid}")
            ids.add(cid)
            stages.add(stage)
            statuses[row["status"]] += 1
            if row["status"] != "completed":
                unfinished.append({"call_uuid": cid, "stage_subdir": stage, "status": row["status"],
                                   "pending_response_bytes": row["pending_response_bytes"]})
            for attempt in row["attempts"]:
                if attempt["usage_known"]:
                    known_attempts += 1
                    known_v2 += score_single_call_usage(
                        prompt_tokens=attempt["prompt_tokens"], cached_tokens=attempt["cached_tokens"],
                        completion_tokens=attempt["completion_tokens"],
                    )["proxy_v2_u_010c_4o"]
                else:
                    unknown_attempts += 1
    return {
        "run_id": prefix[:-2], "status": "partially_recovered_raw_provider_receipts",
        "archive": archive.name, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "stage_directories": len(stages), "unique_logical_calls": len(ids), "status_counts": dict(statuses),
        "known_physical_attempts": known_attempts, "known_physical_v2": known_v2,
        "unclosed_call_count": len(unfinished), "unclosed_calls": unfinished,
        "closed_attempts_with_unknown_usage": unknown_attempts,
        "observed_campaign_total_v2_lower_bound": sealed_known_v2 + known_v2,
        "total_rd_v2_formula": f"{sealed_known_v2 + known_v2} + x + y",
        "y_definition": "Remaining unreported cancelled-run usage. In-progress zero placeholders supply no final cost.",
        "confirmation_seeds_touched": len({name for name in stages if "_m99_c" in name}),
    }


def _historical_withdrawal_links(streams: list[dict]) -> list[dict]:
    rows = []
    for stream in streams:
        receipts = stream["separability_withdrawal_receipts"]
        if not receipts:
            continue
        root = M99_CONF_DIR / "continual_streams" / stream["world_id"] / stream["arm"]
        for rec in receipts:
            m99.SeparabilityWithdrawalReceipt.from_dict(rec)
            matches = []
            for ep_path in root.glob("*/00_episode_summary.json"):
                ep = json.loads(ep_path.read_bytes())
                trial = ep.get("pre_llm_trial_receipt") or {}
                if (trial.get("separability_withdrawal_receipt") or {}).get("receipt_id") == rec["receipt_id"]:
                    matches.append((ep_path, ep, trial))
            if len(matches) != 1:
                raise ValueError(f"Ambiguous historical withdrawal {rec['receipt_id']}")
            path, ep, trial = matches[0]
            step = int(ep["stream_step"])
            before, after = [json.loads((root / f"graph_ckpt_after_step_{n:02d}.json").read_bytes())["questions"]
                             for n in (step - 1, step)]
            def witnesses(q):
                return [w for lineage in q[m96.M96_LINEAGE_FOREST_KEY]["streams"].values() for w in lineage["witness_ledger"]]
            bindings = after["__m9_4_mechanism_executable_bindings__"]["bindings"]
            w_after = witnesses(after)
            up = [w for w in w_after if w["probe_id"] == rec["upstream_component_probe_id"]
                  and w["result"] == "passed" and w["binding_id"] == rec["upstream_binding_id"]
                  and w["solver_code_sha256"] == rec["upstream_code_sha256"]
                  and w["canonical_numeric_sha256"] == rec["counterexample_canonical_numeric_sha256"]
                  and w["public_input_sha256"] == rec["counterexample_public_input_sha256"]]
            neg = [w for w in w_after if w["result"] == "failed" and w["binding_id"] == rec["downstream_binding_id"]
                   and w["public_input_sha256"] == rec["counterexample_public_input_sha256"]]
            pos = [w for w in witnesses(before) if w["result"] == "passed" and w["binding_id"] == rec["downstream_binding_id"]]
            def projection(w):
                return tuple(w["public_structural_features"].get(k) for k in m99.M99_DOWNSTREAM_OBSERVABLE_NAMES)
            diag = trial["conflict_locus_diagnosis"]
            gate = trial["probe_result"]
            checks = {
                "receipt_equals_sealed_episode": rec == trial["separability_withdrawal_receipt"],
                "upstream_ref": len(up) == 1 and diag["probe_id"] == rec["upstream_component_probe_id"]
                    and diag["execution_status"] == "completed" and diag["candidate_code_sha256"] == rec["upstream_code_sha256"],
                "registry_hashes_match": all(hashlib.sha256(bindings[rec[f"{component}_binding_id"]]["canonical_algorithm_code"].encode()).hexdigest()
                                              == rec[f"{component}_code_sha256"] for component in ("upstream", "downstream")),
                "recorded_joint_candidate_and_label_match": gate["candidate_raw_sha256"] == rec["composed_candidate_sha256"]
                    and trial["failure_kind"] == rec["joint_failure_kind"] and gate.get("accepted") is False,
                "downstream_collision": any(projection(a) == projection(b) for a in pos for b in neg),
            }
            rows.append({"world_id": stream["world_id"], "receipt_id": rec["receipt_id"], "episode": str(path.relative_to(M99_CONF_DIR)),
                         "historical_signed_scientific_receipt": bool((gate.get("scientific_probe_receipt") or {}).get("signed_by_host_probe_executor")),
                         "checks": checks, "matches": all(checks.values())})
    return rows


def _replay_historical_withdrawals() -> dict:
    conf = json.loads((M99_CONF_DIR / "m9_9_continual_summary.json").read_bytes())
    links = _historical_withdrawal_links(conf["streams"])
    specs = {spec.world_id: spec for spec in m99.CONFIRMATION_M99_WORLD_SPECS}

    def replay(link):
        path = M99_CONF_DIR / link["episode"]
        ep = json.loads(path.read_bytes())
        step = int(ep["stream_step"])
        hist = ep["pre_llm_trial_receipt"]
        rec = hist["separability_withdrawal_receipt"]
        spec = specs[link["world_id"]]
        compat = spec.to_m98_compat_spec()
        _, tasks, _, logs = m99.build_m99_world_tasks(spec)
        public, host_sha = m98.materialize_m98_public_item_and_sha(tasks[step-1])
        graph = m8p.PersistentMechanismGraph.load_checkpoint(path.parents[1] / f"graph_ckpt_after_step_{step-1:02d}.json")
        registry = m94.M94BindingRegistry(graph)
        m99.update_m99_calibration_snapshot_and_graph(spec=compat, episode_id="audit_current_observation",
                                                      public_item=public, graph=graph, registry=registry)
        up, dn = [registry.bindings[rec[f"{kind}_binding_id"]] for kind in ("upstream", "downstream")]
        cand = m98.compose_m98_two_stage_solver_code(
            upstream_code=up.canonical_algorithm_code, upstream_entrypoint=up.canonical_entrypoint,
            downstream_code=dn.canonical_algorithm_code, downstream_entrypoint=dn.canonical_entrypoint,
            param_payload=m98.assemble_m98_graph_diagonal_payload(compat, graph))
        deps = {"/tmp/hegel_continual_archive/archive_apparatus_log.json": json.dumps(logs[:step+1], indent=2)}
        dep_shas = {key: hashlib.sha256(value.encode()).hexdigest() for key, value in deps.items()}
        cand_sha = hashlib.sha256(cand.encode()).hexdigest()
        if (cand_sha != rec["composed_candidate_sha256"] or host_sha != rec["counterexample_public_input_sha256"]
                or dep_shas != hist["probe_result"]["readonly_dependencies_sha256"]
                or m96.compute_canonical_numeric_item_sha256(public) != rec["counterexample_canonical_numeric_sha256"]):
            raise ValueError(f"Sealed withdrawal candidate, public data or dependencies disagree at {path}")
        sealed_input = ep["public_setup_receipt"]["files_sha256"]["/holdout_task/public_batch.pkl"]
        start = monotonic()
        try:
            gate = m95.execute_single_pre_llm_public_gate_probe_in_container(
                task=tasks[step-1], candidate_code=cand, readonly_dependencies=deps, expected_dependency_sha256=dep_shas)
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            gate = {"setup_ok": False, "infrastructure_exception": str(exc)}
        actual = (gate.get("public_setup_files_sha256") or {}).get("/holdout_task/public_batch.pkl")
        typed = m96.classify_public_probe_failure(gate)
        sci = gate.get("scientific_probe_receipt") or {}
        bound = (actual == sealed_input and gate.get("candidate_raw_sha256") == cand_sha
                 and gate.get("readonly_dependencies_sha256") == dep_shas
                 and gate.get("candidate_bundle_sha256") == hist["probe_result"]["candidate_bundle_sha256"]
                 and sci.get("candidate_bundle_sha256") == gate.get("candidate_bundle_sha256")
                 and sci.get("public_input_digest") == actual)
        return {"world_id": spec.world_id, "historical_receipt_id": rec["receipt_id"],
                "sealed_episode": link["episode"], "sealed_episode_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sealed_joint_scientific_receipt": hist["probe_result"].get("scientific_probe_receipt"),
                "historical_joint_failure_kind": rec["joint_failure_kind"], "sealed_container_public_input_sha256": sealed_input,
                "composed_candidate_sha256": cand_sha, "host_public_pickle_sha256": host_sha,
                "replayed_failure_tier": typed["tier"], "artifact_identity_verified": bound,
                "replayed_bound_scientific_negative": bool(bound and typed["tier"] == "SCIENTIFIC_FAILURE"
                    and _bound_host_measurement(gate, actual) and sci.get("scientific_invariants_verified") is False),
                "probe_wall_ms": round((monotonic()-start)*1000, 2), "public_gate_receipt": gate}

    with ThreadPoolExecutor(max_workers=7) as pool:
        rows = list(pool.map(replay, links))
    return {"schema": "hegel_m991_retrospective_e5_joint_withdrawal_replay_v2", "sandbox": _sandbox_identity(), "paid_llm_calls": 0,
            "executed_at_utc": datetime.now(timezone.utc).isoformat(), "total_receipts": len(rows),
            "historical_signed_scientific_receipts": sum(row["historical_signed_scientific_receipt"] for row in links),
            "replayed_bound_scientific_negatives": sum(row["replayed_bound_scientific_negative"] for row in rows),
            "rows": rows}


def _select_plain_binding(rows: list[dict], chain: str, keys: list[str], features: dict) -> dict:
    candidates = sorted(
        (row for row in rows if row["chain_id"] == chain and row["qualification_status"] == "verified"
         and sorted(row["output_schema_keys"]) == sorted(keys)),
        key=lambda row: (-int(row["selection_priority"]), row["binding_id"]),
    )
    for row in candidates:
        guard = row.get("applicability_guard") or row.get("precondition_guard")
        if m96.evaluate_m96_autonomous_guard(guard, features, allow_open_feature_keys=True)["state"] == "TRUE":
            return row
    raise ValueError(f"No applicable verified plain row for {chain}")


def _build_provenance_and_sensitivity_ledger(cancelled_stage_root: Path | None = None) -> dict[str, Any]:
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
    base_diffs = np.asarray(conf_sum["paired_cluster_inference"]["Hegel_minus_Flat_PlainModuleRegistry_Online"]["cluster_v2_diffs"], dtype=float)
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
    parity_rows = []
    for spec in m99.CONFIRMATION_M99_WORLD_SPECS:
        wid = spec.world_id
        compat = spec.to_m98_compat_spec()
        _, tasks, manifest, logs = m99.build_m99_world_tasks(spec)
        ck5_raw = json.loads((M99_CONF_DIR / "continual_streams" / wid / "Hegel_AutonomousInteractionGraph_Online/graph_ckpt_after_step_05.json").read_text(encoding="utf-8"))
        b_dict = ck5_raw["questions"]["__m9_4_mechanism_executable_bindings__"]["bindings"]
        mechanisms = ck5_raw["questions"]["__m8_authoritative_persistent_state__"]["mechanisms"]
        flat_snap = {"gains": [1.0] * 4, "offsets": [0.0] * 4}
        for node, channels in (("cal_node_p0", spec.channel_perm[:2]), ("cal_node_p1", spec.channel_perm[2:])):
            weights = mechanisms[node]["shared_weights"]
            for ch in channels:
                flat_snap["gains"][ch] = float(weights[f"g_ch{ch}"])
                flat_snap["offsets"][ch] = float(weights[f"b_ch{ch}"])
        for ev_nm in ("E6_separable_revisit", "E7_coupled_revisit"):
            idx = next(i for i, m in enumerate(manifest) if m["causal_event"] == ev_nm)
            pi, psha = m98.materialize_m98_public_item_and_sha(tasks[idx])
            snap_ev = m99.update_m99_calibration_snapshot_and_graph(spec=compat, episode_id=f"{wid}_{ev_nm}", public_item=pi, flat_snapshot=deepcopy(flat_snap))["flat_snapshot"]
            up_obs = m98.compute_m98_upstream_observables(spec.family, pi)
            dn_obs = m98.compute_m98_downstream_observables(logs[: idx + 2])
            comb_obs = {**up_obs, **dn_obs}
            sel_u = _select_plain_binding(list(b_dict.values()), f"{spec.family}_up", ["calibrated_input"], up_obs)
            sel_d = _select_plain_binding(list(b_dict.values()), f"{spec.family}_dn",
                                          m98.expected_m98_public_output_keys(spec.family), comb_obs)
            cand = m98.compose_m98_two_stage_solver_code(
                upstream_code=sel_u["canonical_algorithm_code"],
                upstream_entrypoint=sel_u["canonical_entrypoint"],
                downstream_code=sel_d["canonical_algorithm_code"],
                downstream_entrypoint=sel_d["canonical_entrypoint"],
                param_payload=snap_ev,
            )
            csha = hashlib.sha256(cand.encode("utf-8")).hexdigest()
            ref = hg_abl_rows[(wid, ev_nm)]
            sealed_docker_sha = _sealed_container_public_sha(wid, idx + 1)
            parity_rows.append({
                "world_id": wid, "event": ev_nm, "selected_upstream": sel_u["binding_id"],
                "selected_downstream": sel_d["binding_id"], "candidate_sha256": csha, "public_pickle_sha256": psha,
                "policy_input_source": "historical_host_reconstruction",
                "historical_docker_public_input_sha256": sealed_docker_sha,
                "policy_input_matches_historical_docker_pickle": psha == sealed_docker_sha,
                "matched": (sel_u["binding_id"] == ref["selected_upstream"]
                            and sel_d["binding_id"] == ref["selected_downstream"]
                            and csha == ref["candidate_sha256"] and psha == ref["public_batch_pkl_sha256"]),
            })

    dev_v2 = sum(row["proxy_v2_u_010c_4o"] for row in dev_calls)
    conf_v2 = sum(row["proxy_v2_u_010c_4o"] for row in conf_calls)
    cancelled = _cancelled_campaign_ledger(cancelled_stage_root, {row["call_uuid"] for row in dev_calls + conf_calls}, dev_v2 + conf_v2)
    finalizer = _compute_commit_py_tree_sha256("d4b1cd7f0", omit_report_generator=True)
    if finalizer["source_tree_sha256"] != conf_sum["source_tree_sha256_at_start"]:
        raise ValueError("Reconstructed post-freeze finalizer differs from sealed tree")
    withdrawal_rows = _historical_withdrawal_links(conf_sum["streams"])
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
            "finalizer_reconstruction_excluding_later_report_generator": finalizer,
            "finalizer_digest_matches_recorded": True,
            "paid_start_worktree_state_independently_verified": False,
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
            "sealed_dev_physical_v2": dev_v2,
            "sealed_confirmation_physical_v2_known": conf_v2,
            "combined_sealed_physical_v2_formula": f"{dev_v2 + conf_v2} + x",
            "cancelled_dev_diagnostic_run": cancelled,
        },
        "historical_withdrawal_foreign_keys": {"matched_receipts": sum(r["matches"] for r in withdrawal_rows),
                                              "total_receipts": len(withdrawal_rows),
                                              "historical_typed_scientific_receipts": sum(r["historical_signed_scientific_receipt"] for r in withdrawal_rows),
                                              "rows": withdrawal_rows},
        "unknown_x_exact_sign_flip_sensitivity_ladder": {
            "call_uuid": "0391d702b0c547a488608847b5dff224",
            "arm": "Flat_PlainModuleRegistry_Online",
            "location": "alpha_m99_c04 / E1_down / attempt 1 timeout",
            "threshold_for_two_sided_p_lt_005": "x > 53450",
            "evaluated_points": x_ladder,
        },
        "plain_dict_same_joint_features_step05_replay": {
            "matched_episodes": sum(row["matched"] for row in parity_rows),
            "total_episodes": len(parity_rows),
            "policy_inputs_compared_to_historical_docker": len(parity_rows),
            "policy_inputs_matching_historical_docker": sum(row["policy_input_matches_historical_docker_pickle"] for row in parity_rows),
            "rows": parity_rows,
            "attribution_scope": "Post-hoc execution of frozen step-05 rows with all current joint features available to the plain downstream selector. The control isolates selection and composition conditional on the same learned bindings and parameters. Original online learning costs and discovery are outside this control.",
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


def _compare_public_materializations(q01: dict, withdrawals: dict) -> dict:
    """Compare host and sandbox values on each unique public task behind the two replays."""
    task_ids = {r["target_task_id"] for r in q01["probes"]}
    task_ids.update(Path(r["sealed_episode"]).parent.name for r in withdrawals["rows"])
    selected = []
    for spec in m99.CONFIRMATION_M99_WORLD_SPECS:
        source, tasks, _, _ = m99.build_m99_world_tasks(spec)
        selected.extend(t for t in (source, *tasks) if t.task_id in task_ids)
    if {task.task_id for task in selected} != task_ids:
        raise ValueError("Unresolved public task in the historical replays")

    def leaves(value, path=()):
        if isinstance(value, (tuple, list, dict)):
            items = value.items() if isinstance(value, dict) else enumerate(value)
            for key, child in items:
                yield from leaves(child, (*path, str(key)))
        else:
            yield path, np.asarray(value)

    def compare(task):
        host, host_sha = m98.materialize_m98_public_item_and_sha(task)
        sandbox, sandbox_sha = m99.materialize_m99_public_item_and_sha(task)
        a, b = dict(leaves(host)), dict(leaves(sandbox))
        schema_diff = len(a.keys() ^ b.keys())
        masks = discrete = 0
        largest, location = 0.0, None
        for key in sorted(a.keys() & b.keys()):
            x, y = a[key], b[key]
            if x.shape != y.shape or x.dtype != y.dtype:
                schema_diff += 1
            elif x.dtype.kind == "f":
                finite = np.isfinite(x) & np.isfinite(y)
                masks += int(any(not np.array_equal(mask(x), mask(y))
                                 for mask in (np.isnan, np.isposinf, np.isneginf)))
                if np.any(finite):
                    diff = float(np.max(np.abs(x[finite] - y[finite])))
                    if diff > largest:
                        largest, location = diff, list(key)
            elif not np.array_equal(x, y):
                discrete += 1
        return dict(task_id=task.task_id, host_pickle_sha256=host_sha,
                    sandbox_pickle_sha256=sandbox_sha, schema_differences=schema_diff,
                    nonfinite_mask_differences=masks, discrete_field_differences=discrete,
                    max_abs_float_difference=largest, max_difference_path=location)

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(compare, sorted(selected, key=lambda task: task.task_id)))
    return dict(scope="Unique exposed public tasks used for Q01 and the seven E5 replays",
                paid_llm_calls=0, distinct_tasks=len(rows), rows=rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cancelled-stage-root", type=Path, help="Original provider stage root used only on first recovery")
    parser.add_argument("--refresh-probes", action="store_true", help="Rerun Docker evidence and replace the M9.9.1 receipts")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p01 = OUT_DIR / "01_m99_q01_30_cross_diagonal_decomposition_probes.json"
    existing_q01 = json.loads(p01.read_bytes()) if p01.is_file() else {}
    if (not args.refresh_probes and existing_q01.get("schema") == "hegel_m9_9_1_q01_30_cross_diagonal_backverification_v2"
            and existing_q01.get("infra_or_timeout_count") == 0):
        print("[M9.9.1] 1/3 Reusing sealed 30 off-diagonal Docker public-gate back-verification probes (`Q01`)...", flush=True)
        r01 = existing_q01
    else:
        print("[M9.9.1] 1/3 Running 30 off-diagonal Docker public-gate back-verification probes (`Q01`)...", flush=True)
        r01 = _run_30_cross_diagonal_probes()
        p01.write_text(json.dumps(r01, indent=2) + "\n", encoding="utf-8")

    p02 = OUT_DIR / "02_m99_q03_c04_e7_existing_candidate_container_verification.json"
    if not args.refresh_probes and p02.exists() and json.loads(p02.read_bytes()).get("schema") == "hegel_m9_9_1_q03_c04_e7_fallback_container_verification_v2":
        print("[M9.9.1] 2/3 Reusing sealed `alpha_m99_c04/E7` existing `up_b02` container verification (`Q03`)...", flush=True)
        r02 = json.loads(p02.read_text(encoding="utf-8"))
    else:
        print("[M9.9.1] 2/3 Running `alpha_m99_c04/E7` existing `up_b02` container verification (`Q03`)...", flush=True)
        r02 = _run_c04_e7_existing_candidate_container_verification()
        p02.write_text(json.dumps(r02, indent=2) + "\n", encoding="utf-8")

    print("[M9.9.1] 3/3 Building provenance, x sensitivity, cancelled-run usage and plain-dict replay ledger...", flush=True)
    r03 = _build_provenance_and_sensitivity_ledger(args.cancelled_stage_root)
    (OUT_DIR / "03_m99_provenance_x_sensitivity_and_plain_dict_replay.json").write_text(json.dumps(r03, indent=2) + "\n", encoding="utf-8")

    p06 = OUT_DIR / "06_m99_q04_original_withdrawal_joint_replay.json"
    existing_q06 = json.loads(p06.read_bytes()) if p06.is_file() else {}
    if args.refresh_probes or existing_q06.get("schema") != "hegel_m991_retrospective_e5_joint_withdrawal_replay_v2":
        print("[M9.9.1] Replaying sealed withdrawal candidates on their original E5 public batches...", flush=True)
        r06 = _replay_historical_withdrawals()
        p06.write_text(json.dumps(r06, indent=2) + "\n")
    else:
        r06 = existing_q06

    # Preserve both identities: host generation and the actual Docker batch can differ numerically.
    for probe in r01["probes"]:
        probe["host_canonical_numeric_sha256"] = probe["canonical_numeric_sha256"]
        probe["executed_canonical_numeric_sha256"] = (probe["public_gate_receipt"].get("scientific_probe_receipt") or {}).get("canonical_numeric_sha256")
    for row in r06["rows"]:
        sealed_ep = json.loads((M99_CONF_DIR / row["sealed_episode"]).read_bytes())
        rec = sealed_ep["pre_llm_trial_receipt"]["separability_withdrawal_receipt"]
        row["host_canonical_numeric_sha256"] = rec["counterexample_canonical_numeric_sha256"]
        row["executed_canonical_numeric_sha256"] = (row["public_gate_receipt"].get("scientific_probe_receipt") or {}).get("canonical_numeric_sha256")
    for path, ledger, rows in ((p01, r01, r01["probes"]), (p06, r06, r06["rows"])):
        measured = [row for row in rows if row["executed_canonical_numeric_sha256"]]
        ledger["host_vs_executed_numeric_identity"] = {
            "compared": len(measured),
            "matched": sum(row["host_canonical_numeric_sha256"] == row["executed_canonical_numeric_sha256"] for row in measured),
            "historical_execution_reference": "Sealed Docker public_setup_receipt.files_sha256 for the same public batch",
        }
        output = json.dumps(ledger, indent=2) + "\n"
        if path.read_text() != output:
            path.write_text(output)

    corrected_ablation_path = OUT_DIR / "07_m99_missing_prefix_ablation.json"
    if args.refresh_probes or not corrected_ablation_path.is_file():
        import run_m9_9_autonomous_decomposition_experiment as runner

        specs = runner.CONFIRMATION_M99_WORLD_SPECS
        worlds = {spec.world_id: runner.build_m99_world_tasks(spec) for spec in specs}
        ablation = runner.run_m99_factorized_ceiling_ablation_on_sealed_streams(
            world_specs=specs, built_worlds=worlds, out_dir=M99_CONF_DIR)
        ablation["evidence_scope"] = "Retrospective regrade with explicit missing-prefix abstention"
        ablation["source_ablation_sha256"] = hashlib.sha256(
            (M99_CONF_DIR / "m9_9_factorized_ceiling_ablation.json").read_bytes()).hexdigest()
        corrected_ablation_path.write_text(json.dumps(ablation, indent=2) + "\n", encoding="utf-8")

    comparison_path = OUT_DIR / "08_m99_host_sandbox_public_values.json"
    if args.refresh_probes or not comparison_path.is_file():
        comparison = _compare_public_materializations(r01, r06)
        comparison_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")

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
                "q04_historical_typed_receipts": r06["historical_signed_scientific_receipts"],
                "q04_replayed_bound_scientific_negatives": r06["replayed_bound_scientific_negatives"],
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
