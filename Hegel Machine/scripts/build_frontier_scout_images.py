#!/usr/bin/env python3
"""Build lightweight official Docker images + offline verifier wheels for the 20-task Frontier Scout Benchmark."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys

WORKTREE_ROOT = Path(__file__).resolve().parents[2]
EXT_ROOT = WORKTREE_ROOT / "external_benchmarks"
TB_TASKS_ROOT = EXT_ROOT / "terminal-bench-science" / "tasks"
BM_ROOT = EXT_ROOT / "bio-mystery-bench"
WHEELS_DIR = WORKTREE_ROOT / ".verifier_wheels"

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
    r = subprocess.run(["docker", "image", "inspect", tag], capture_output=True)
    return r.returncode == 0


def _build_one_tb(item: tuple[str, str]) -> tuple[str, bool, str]:
    name, rel = item
    tag = f"tb_sci_{name}:v01"
    if _img_exists(tag):
        return name, True, "cached"
    env_dir = TB_TASKS_ROOT / rel / "environment"
    proc = subprocess.run(
        ["docker", "build", "-t", tag, str(env_dir)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        return name, False, proc.stderr[-500:]
    return name, True, "built"


def build_biomystery_slim() -> tuple[bool, str]:
    tag = "hegel_biomystery_slim:v1"
    if _img_exists(tag):
        return True, "cached"
    df_content = """FROM python:3.11-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \\
    curl wget ca-certificates procps gzip bzip2 xz-utils unzip \\
    samtools bcftools bedtools ncbi-blast+ \\
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \\
    numpy scipy pandas scikit-learn biopython requests matplotlib seaborn h5py pyarrow statsmodels
RUN mkdir -p /workspace/data /workspace/scratch /workspace/output /workspace/skills
WORKDIR /workspace
COPY SKILLS/deg-functional-enrichment/SKILL.md /workspace/skills/deg-functional-enrichment.md
COPY SKILLS/chipseq-tf-identification/SKILL.md /workspace/skills/chipseq-tf-identification.md
COPY SKILLS/genome-retrieval/SKILL.md /workspace/skills/genome-retrieval.md
COPY SKILLS/ucsc-sequence-fetch/SKILL.md /workspace/skills/ucsc-sequence-fetch.md
COPY SKILLS/blast-search/SKILL.md /workspace/skills/blast-search.md
"""
    slim_df = BM_ROOT / "docker" / "Dockerfile.slim"
    slim_df.write_text(df_content, encoding="utf-8")
    proc = subprocess.run(
        ["docker", "build", "-t", tag, "-f", str(slim_df), str(BM_ROOT)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        return False, proc.stderr[-500:]
    return True, "built"


def main() -> int:
    WHEELS_DIR.mkdir(parents=True, exist_ok=True)
    if not list(WHEELS_DIR.glob("pytest_json_ctrf*")):
        print("Downloading offline verifier wheels (pytest + pytest-json-ctrf)...", flush=True)
        subprocess.run(
            [
                sys.executable, "-m", "pip", "download",
                "--dest", str(WHEELS_DIR),
                "pytest==8.4.1", "pytest-json-ctrf==0.3.5",
            ],
            check=True,
        )

    print("Building BioMysteryBench slim container...", flush=True)
    bm_ok, bm_msg = build_biomystery_slim()
    print(f"  [BioMystery] ok={bm_ok} ({bm_msg})", flush=True)
    if not bm_ok:
        return 1

    print("Building 15 Terminal-Bench-Science v0.1.0 task images (4 parallel builders)...", flush=True)
    failures = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(_build_one_tb, it) for it in TB_15_TASKS]
        for fut in as_completed(futs):
            name, ok, msg = fut.result()
            print(f"  [TB-Sci] {name:36s} ok={ok} ({msg})", flush=True)
            if not ok:
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
