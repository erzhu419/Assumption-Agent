# SkillLearnBench Network Isolation

## Incident finding

The benchmark dataset and verifiers are local, but the upstream task runner used
ordinary Docker egress. Task agents consequently ran `pip`, `apt`, `npm`, Maven,
GitHub, and Hugging Face downloads during evaluation. Local benchmark storage did
not make the task runtime offline.

## Evaluation mode

Paper evaluation is fail-closed:

- task images and the Codex runtime must already exist locally;
- Docker image pulls and online dependency builds are disabled;
- each task container uses `assumption-v2-restricted`;
- external DNS is disabled and `ruoli.dev` is pinned in `/etc/hosts`;
- the host `DOCKER-USER` chain permits only the pinned model endpoint on TCP 443;
- a Docker network-I/O watchdog stops any trial above 32 MiB total traffic;
- all other container egress, including PyPI, GitHub, Hugging Face, Maven, and
  Ubuntu repositories, is rejected.

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

The 2026-07-11 local probe resolved the host to `101.6.15.130` and downloaded the
11,053-byte `six==1.16.0` wheel entirely from the same host. Router policy should
prefer a domain-based direct-route rule because the address may change.
