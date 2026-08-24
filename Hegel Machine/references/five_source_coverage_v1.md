# Hegel Machine five-source coverage v1

This note describes the additive coverage ledger in
`five_source_coverage_manifest_v1.json`. It does not replace or modify
`manifest.json`, `checksums.sha256`, any source Markdown, or any archived
payload. No network access or bundle copying was used.

The canonical coverage-manifest bytes are 60,336 bytes with SHA-256
`75fcdba4993bb3e792a3993db25db5f91b8c408ba83abbcb873dff62ea4bbb65`.

## Source boundary

The authoritative copies for this audit are the five files under
`Hegel Machine/markdown/`. Their hashes match the existing source checksum
ledger. Three legacy originals remain byte-identical. The two legacy paths
`reconstruction_v2/markdown/黑格尔机.md` and
`reconstruction_v2/markdown/黑格尔机和泛函分析.md` are absent, so they are
recorded as missing rather than treated as live sources.

Across the five authoritative files, the ledger resolves 52 distinct URL
identities after removing only `utm_*`, normalizing arXiv aliases, and
preserving other identity-bearing query parameters. The dispositions are:

- 18 archived identities backed by the in-tree Hegel reference manifest;
- 2 in-tree access-limited identities (Cambridge and Wiley);
- 28 identities backed only by external local bundles;
- 2 OpenAI identities with honest in-tree challenge records plus external
  local rendered snapshots;
- 2 project provenance pointers, recorded but not duplicated as archives.

There are no unrouted source URL identities. This is a coverage statement,
not a self-containment statement.

## Self-containment and limitations

`Hegel Machine/references` is not a standalone five-source archive. The
roadmap and GSCL payload roots recorded in the JSON are outside this worktree
and are explicitly classified `local_external_not_self_contained`. Their
observed file hashes and repository commit/tree identities are frozen in the
ledger, but a clone of this worktree will not contain those external bytes.

The direct OpenAI fetches remain access-limited even though local rendered
snapshots exist. The Cambridge chapter and Wiley AGM paper remain
access-limited. The anonymous `number_game-F153` repository remains incomplete
and non-reconstructable; it is not a verified repository archive.

`reference/repos_manifest.md` is retained only as historical text. Its claimed
`reference/repos` directory is absent in this worktree, so it is explicitly
deprecated as current repository-existence evidence.

## Integrity scope

The old Hegel checksum ledger continues to cover its original 135 entries and
does not cover these two additive closure files. The JSON instead binds the
observed source files, legacy ledgers, external artifacts, portable manifests,
and external repository commits/trees. This note records the JSON's final
digest. Neither file asserts that an external local bundle is portable or
in-tree.
