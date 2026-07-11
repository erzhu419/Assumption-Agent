# Network Incident Audit: 2026-07-11

## Containment

- The active SkillLearnBench development run was interrupted.
- All `evaluation_*` containers were removed; no evaluation container remains.
- Local clones of `cc-llm4zotero-adapter` and `llm-for-zotero` were deleted.
- No Zotero bridge process, Docker image, volume, or container was present.
- Existing task images were retained because stopped images generate no traffic and
  deleting them would force another large dependency download.

## Attribution

Router and local connection evidence attributed the largest paid route to
`151.101.0.223:443` (PyPI/Fastly) from SkillLearnBench `evaluation_*` containers.
The timing followed the evaluation start and included approximately 8.97 GB of
PyPI traffic. GitHub, Ubuntu, Maven, and Hugging Face traffic also aligned with
the same task run.

An offline scan of 102 container `codex.txt` logs found:

- 65 `pip install` lines;
- 36 `apt update` or `apt install` lines;
- 22 `npm` or `npx` lines;
- 19 Maven command lines;
- 17 `curl` lines and 8 `wget` lines;
- 5,813 Maven Central URL traces;
- 1,925 GitHub URL traces;
- 305 Hugging Face URL traces;
- 132 Ubuntu repository URL traces.

The benchmark data was local. The missing control was task-runtime egress: the
upstream runner allowed agents and verifiers to install or fetch resources while
answering a local benchmark task.

The Zotero bridge was not on the active route. Evaluation containers invoked the
OpenAI-compatible Codex provider directly at `https://ruoli.dev/v1`. WSL had no
proxy variables or local policy routing; it had one default gateway, so the
Windows/router policy decided whether each public destination used the VPN.

## Model Traffic

Before interruption, 91 completed trial events reported about 70.4 million total
tokens. Median usage was about 312 thousand tokens, p90 about 1.98 million, and
the maximum about 7.13 million. Therefore dependency downloads were the largest
identified source, but repeated model context upload was also a material residual
cost and required its own limit.

## Remediation

Paper evaluation now has four fail-closed controls:

1. `fail_closed_prebuilt_only_v1`: missing task images or Codex runtime volumes
   stop the run before any build or install.
2. `docker_user_endpoint_allowlist_v1`: a dedicated `172.29.0.0/24` Docker
   network can reach only pinned `ruoli.dev` IPv4 addresses on TCP 443.
3. `pinned_hosts_no_external_dns_v1`: external DNS is disabled, IPv6 is disabled,
   and the provider hostname is injected through `/etc/hosts`.
4. `docker_stats_hard_byte_cap_v1`: each trial is stopped above 32 MiB total
   network I/O and emits start, exceeded, and final-usage audit events.

The Codex runtime also disables analytics and OpenTelemetry export. Evaluation
containers receive offline settings for pip, uv, npm, Hugging Face, Transformers,
Cargo, and Git prompts, and every Docker run uses `--pull=never`.

## Verification

- PyPI/Fastly `151.101.0.223:443`: blocked from the restricted network.
- Tsinghua TUNA `101.6.15.130:443`: blocked from the restricted network.
- Public DNS lookup for `pypi.org`: failed with `SERVFAIL`.
- Pinned provider `45.78.76.197:443`: TCP connection allowed; no API request was
  sent during the isolation probe.
- Firewall counters recorded both accepted provider packets and rejected packets.
- A real idle-container monitor lifecycle recorded 862 bytes and finalized cleanly.
- Train plus validation cache audit: 60/60 items passed, 38 unique images,
  zero builds, zero installs, and `online_build_attempted=false`.
- Full local test suite passed and paper preflight reported no blockers.

## Direct Dependency Preparation

Python dependency preparation is separate from evaluation. The configured mirror
is `https://pypi.tuna.tsinghua.edu.cn/simple`. A local end-to-end probe resolved
`pypi.tuna.tsinghua.edu.cn` to `101.6.15.130` and downloaded the 11,053-byte
`six==1.16.0` wheel from that same host. Router rules should prefer the hostname
because the address may rotate.

## Restart Gate

No full run should restart immediately. The next live step is one non-claim train
canary with one worker. It must show:

- no non-provider egress;
- network usage below the per-trial cap;
- no image build, pull, or online package install;
- no residual container after completion;
- explicit accounting of token usage, bytes, duration, and error type.

Only after that canary should a two-item paired smoke test run. Sealed test data
remains untouched.
