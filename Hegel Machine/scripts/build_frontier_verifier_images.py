#!/usr/bin/env python3
"""Build the 15 official Harbor Separate-Verifier Docker images (`tb_sci_verifier_<name>:v01`) + BioMystery extras."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess

WORKTREE_ROOT = Path(__file__).resolve().parents[2]
EXT_ROOT = WORKTREE_ROOT / "external_benchmarks"
TB_TASKS_ROOT = EXT_ROOT / "terminal-bench-science" / "tasks"
BM_ROOT = EXT_ROOT / "bio-mystery-bench"

TB_15_TASKS = [
    ("ode-law-discovery", "mathematical-sciences/applied-mathematics/ode-law-discovery"),
    ("traffic-flux-inversion", "mathematical-sciences/applied-mathematics/traffic-flux-inversion"),
    ("amr-poisson-optimize", "mathematical-sciences/applied-mathematics/amr-poisson-optimize"),
    ("baseline-free-localization", "engineering-sciences/mechanical-engineering/baseline-free-localization"),
    ("reactor-safety-control", "engineering-sciences/chemical-engineering/reactor-safety-control"),
    ("navigation-sensor-calibration", "engineering-sciences/electrical-engineering/navigation-sensor-calibration"),
    ("sparse-network-assimilation", "earth-sciences/atmospheric-sciences/sparse-network-assimilation"),
    ("nanoindentation-property-extraction", "physical-sciences/materials-science/nanoindentation-property-extraction"),
    ("symbolic-regression", "mathematical-sciences/statistics/symbolic-regression"),
    ("koopman-mfg-id", "mathematical-sciences/applied-mathematics/koopman-mfg-id"),
    ("certified-sparse-regression", "mathematical-sciences/operations-research/certified-sparse-regression"),
    ("guided-wave-localization", "engineering-sciences/mechanical-engineering/guided-wave-localization"),
    ("virtual-baseline-localization", "engineering-sciences/mechanical-engineering/virtual-baseline-localization"),
    ("spin-glass-groundstate", "physical-sciences/physics/spin-glass-groundstate"),
    ("clinical-metadata-recovery", "life-sciences/medicine/clinical-metadata-recovery"),
]


def _img_exists(tag: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0


def _build_one_verifier(item: tuple[str, str]) -> tuple[str, bool, str]:
    name, rel = item
    tag = f"tb_sci_verifier_{name}:v01"
    if _img_exists(tag):
        return name, True, "cached"
    tests_dir = TB_TASKS_ROOT / rel / "tests"
    proc = subprocess.run(
        ["docker", "build", "-t", tag, str(tests_dir)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        return name, False, proc.stderr[-600:]
    return name, True, "built"


def _upgrade_biomystery_extras() -> tuple[bool, str]:
    tag = "hegel_biomystery_slim:v2"
    if _img_exists(tag):
        return True, "cached"
    df_txt = """FROM hegel_biomystery_slim:v1
RUN pip install --no-cache-dir gemmi pyjaspar pysam networkx
"""
    df_p = BM_ROOT / "docker" / "Dockerfile.slim_v2"
    df_p.write_text(df_txt, encoding="utf-8")
    proc = subprocess.run(
        ["docker", "build", "-t", tag, "-f", str(df_p), str(BM_ROOT)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        return False, proc.stderr[-500:]
    return True, "built"


def main() -> int:
    print("Adding gemmi/pyjaspar/pysam/networkx layer to BioMysteryBench slim image...", flush=True)
    ok_bm, msg_bm = _upgrade_biomystery_extras()
    print(f"  [BioMystery v2] ok={ok_bm} ({msg_bm})", flush=True)
    if not ok_bm:
        return 1

    print("Building 15 official Harbor Separate-Verifier images (4 parallel builders)...", flush=True)
    failures = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(_build_one_verifier, it) for it in TB_15_TASKS]
        for fut in as_completed(futs):
            name, ok, msg = fut.result()
            print(f"  [TB-Verifier] {name:36s} ok={ok} ({msg})", flush=True)
            if not ok:
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
