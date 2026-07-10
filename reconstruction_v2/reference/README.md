# Local Reference Bundle

This directory intentionally keeps literature and cloned research repositories out of Git.

Local contents:

- `The Red Queen Godel Machine Co-Evolving Agents and Their Evaluators.pdf`
- `self_evo_continual_20260707/`
- `provider_bridges/cc-llm4zotero-adapter/`
- `provider_bridges/llm-for-zotero/`

The self-evolution bundle contains the downloaded papers, metadata, pages, and reference repositories used to design reconstruction v2.

Provider bridge provenance at integration time:

- `cc-llm4zotero-adapter` commit `2ebeec8111480541badd1a1b9e899e5ca630f560` (MIT). Its current runtime is Claude Code-only; reconstruction v2 uses its local bridge/process-isolation pattern as a reference, not as a GPT provider.
- `llm-for-zotero` commit `770a9ec65cdca0bc6e07b5eb8ae6eef3444aad81` (AGPL-3.0). Its recommended ChatGPT Plus path is Codex App Server. Reconstruction v2 independently implements the small public JSON-RPC transport contract and invokes the installed Codex CLI; no AGPL source is copied into the package.
