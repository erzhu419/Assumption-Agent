# Local Reference Bundle

This directory intentionally keeps literature and cloned research repositories out of Git.

Local contents:

- `The Red Queen Godel Machine Co-Evolving Agents and Their Evaluators.pdf`
- `self_evo_continual_20260707/`
- `gpt_advice_roadmap_20260729/`
- `generalized_structural_correspondence_20260730/`
- `gscl_intrinsic_candidates_20260730/`

The self-evolution bundle contains the downloaded papers, metadata, pages, and reference repositories used to design reconstruction v2.

The GPT advice and roadmap bundle contains the papers, web snapshots, arXiv source, verified
author/project repositories, exact paper hashes, and a manifest used for
`docs/gpt_advice_roadmap_transfer_assessment_20260729.md`. The anonymous repository associated
with paper 25 is explicitly marked incomplete in its manifest; the paper itself is present.

The generalized structural correspondence bundle contains the seven papers and five verified
repository lineages used by `docs/gscl_reference_bundle_manifest_20260730.md`. The GSCL
intrinsic-candidate bundle contains the locally acquired public benchmark candidates considered
for the post-qualification untouched measurement. Both bundles remain local research inputs;
their tracked manifests, hashes, and code-facing conclusions are the portable record.

Provider bridge provenance at integration time:

- The temporary Zotero LLM provider bridges were removed after the network-usage audit. Reconstruction v2 now uses the frozen OpenAI-compatible route directly and does not require a Zotero, Docker, or Codex App Server bridge.
