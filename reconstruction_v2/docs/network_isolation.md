# SkillLearnBench Network Isolation

## Incident finding

The benchmark dataset and verifiers are local, but the upstream task runner used
ordinary Docker egress. Task agents consequently ran `pip`, `apt`, `npm`, Maven,
GitHub, and Hugging Face downloads during evaluation. Local benchmark storage did
not make the task runtime offline.

## Evaluation mode

Paper evaluation is fail-closed:

- task images and the Codex runtime must already exist locally;
- verifier dependencies must come from a declared content-addressed local profile;
- Docker image pulls and online dependency builds are disabled;
- each task container uses `assumption-v2-restricted`;
- external DNS is disabled and `ruoli.dev` is pinned in `/etc/hosts`;
- the host `DOCKER-USER` chain permits only the pinned model endpoint on TCP 443;
- a Docker network-I/O watchdog stops any trial above 32 MiB total traffic;
- all other container egress, including PyPI, GitHub, Hugging Face, Maven, and
  Ubuntu repositories, is rejected.
- Codex web search and image generation are disabled; the full JSONL trace is
  audited for remote-tool calls and runtime package-install commands.
- a trial is valid only when a parseable `reward.txt` and non-empty pytest CTRF
  execution receipt agree that the verifier actually ran.

Any task whose verifier contains online installation commands but lacks a local
profile is rejected before the model container starts. This keeps unsupported
families from consuming either provider traffic or package traffic.

The model request remains online and is the only intended evaluation traffic.
No full evaluation should run until the egress denial probe passes.

## Dependency preparation

Dependency preparation is a separate, explicit operation. It is never invoked by
the paper pipeline. For Python packages, source
`scripts/dependency_preparation_env.sh` to select the Tsinghua TUNA mirror:

```bash
source scripts/dependency_preparation_env.sh
```

Frozen mirror endpoint:

```text
https://pypi.tuna.tsinghua.edu.cn/simple
```

Six content-addressed profiles now cover 76 tasks across Python 3.8, 3.10, 3.11,
and 3.12. Together with five network-free dbscan tasks, this gives 81/95
credential-independent tasks a local verifier path. The poster profile pins
pytest, CTRF, Pillow, NumPy, and python-docx; the shared light profiles add only
pytest/CTRF and pypinyin where required. Larger packages such as pandas,
python-docx, PyPDF2, and OpenCV already exist in their immutable task images and
are not downloaded again.

The cached wheelhouses total 40,893,608 bytes: 31,557,639 bytes for the original
poster profile and 9,335,969 bytes for the five additional ABI/profile sets.
Their initial downloads used TUNA. The v2 runtime rebuild reused every
wheelhouse, attempted zero downloads, and installed and probed all six volumes
with Docker `--network none`. Runtime volumes are mounted read-only during
evaluation.

The v2 verifier wrapper also preserves the local semantic/audit behavior that
is independent of package installation: `/root` working directory,
`RESULTS_PATH`, court PDFs, the dependency-audit CSV, and the travel itinerary.

Preparation is explicit and is not called by the evaluation pipeline:

```bash
sg docker -c 'env PYTHONPATH=. python3 -m assumption_agent.benchmarks.offline_verifier \
  --profile anthropic-poster-py312-v1 \
  --base-image-tag assumption-v2-item:d8eaa6ca2b652f13fe2145e7 \
  --report artifacts/offline_verifier_prep/poster.json \
  --events artifacts/offline_verifier_prep/events.jsonl'
```

The train-only verifier matrix is also offline and model-free:

```bash
sg docker -c 'env PYTHONPATH=. python3 -m \
  assumption_agent.benchmarks.offline_verifier_matrix \
  --root reference/self_evo_continual_20260707/repos/SkillLearnBench \
  --manifest manifests/skilllearnbench_instance_holdout_credential_independent_v1.json \
  --output-root artifacts/offline_verifier_matrix \
  --events artifacts/offline_verifier_matrix/events.jsonl'
```

See `docs/skilllearn_offline_verifier_matrix.md` for family coverage. A full
95-task paper run remains blocked on 14 tasks until every selected family is
either localized or excluded by a preregistered infrastructure rule.

The 2026-07-11 local probe resolved the host to `101.6.15.130` and downloaded the
11,053-byte `six==1.16.0` wheel entirely from the same host. Router policy should
prefer a domain-based direct-route rule because the address may change.

## Real model canary

The first model-only poster canary used `gpt-5.4-mini`, the pinned ruoli route,
and the 32 MiB fuse. It completed in 138.82 seconds with 4,399,999 total Docker
network bytes, zero remote-tool calls, zero runtime-install commands, and a valid
five-test CTRF receipt. Raw scored 0 because one expected brand color was wrong;
four of five tests passed. This is a real task failure, not a verifier dependency
failure. The canary is diagnostic-only and is not part of any paper metric.
