# GSCL Reference Bundle Manifest

Date: 2026-07-30  
Bundle root:
`reconstruction_v2/reference/generalized_structural_correspondence_20260730/`  
Source document: `reconstruction_v2/markdown/广义对位关系.md`

The bundle is local and intentionally ignored by Git. This tracked manifest records
the exact sources, payload hashes, repository commits, and verification status.
No online/API evaluator was used.

## Coverage

- Explicit URLs in the source document: 8 unique URLs.
- Papers resolved and acquired: 7/7.
- External repositories with a verified paper/code lineage: 5/5.
- Current Assumption-Agent repository: recorded, not duplicated.
- Unresolved paper identities: 0.
- Paper-specific repositories not found: 3 papers, explicitly recorded below.

The environment did not expose the academic-search MCP described by the local
literature skill. Resolution therefore used primary arXiv, PMLR, NeurIPS, AAAI,
LMCS, PubMed/Elsevier and OSF pages, plus author/paper repository statements.

## Papers

| ID | Local file | Bytes | Pages | SHA256 | Primary source | Code status |
|---|---|---:|---:|---|---|---|
| P01 | `papers/01_hipporag_neurips2024.pdf` | 3,256,579 | 38 | `bf2af160f3e34fb424026a04714c6b1ca61be8b59fc3832ebe3583f221e43f88` | [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/6ddc001d07ca4f319af96a3024f6dbd1-Paper-Conference.pdf) | official repo acquired |
| P02 | `papers/02_functorial_data_migration_arxiv1009.1166.pdf` | 437,144 | 30 | `0b4adddb4cc5701c3ce6dcabaf79367b42134f3baab0d368b9f3a6fcefd70009` | [arXiv 1009.1166](https://arxiv.org/pdf/1009.1166) | archived FQL and maintained CQL lineage acquired |
| P03 | `papers/03_categorical_deep_learning_icml2024.pdf` | 650,970 | 33 | `cc39775337cb88c0624a6ca05c4ffa8f9181dd9c3f0033d4b8ebcc8c7d28e2c6` | [PMLR](https://raw.githubusercontent.com/mlresearch/v235/main/assets/gavranovic24a/gavranovic24a.pdf) | no paper-specific implementation declared |
| P04 | `papers/04_hake_aaai2020.pdf` | 596,705 | 8 | `31e55e763f0d1038e1bfb94fba7b4d23e9c89180535a9327fc16adbd44ef1893` | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/download/5701/5557) | official repo acquired |
| P05 | `papers/05_hypergraph_pagerank_icml2024.pdf` | 4,896,350 | 25 | `b8fba1b783fbadf65098052214089e027d47857988470eebff15547559d49c33` | [PMLR](https://raw.githubusercontent.com/mlresearch/v235/main/assets/ameranis24a/ameranis24a.pdf) | author-linked repo acquired |
| P06 | `papers/06_taxonomy_categories_relations_lmcs2026.pdf` | 705,713 | 38 | `832c0f076ba7609734f4bcdac5a91967bfa9bb53429754a6d44396c4f992be66` | [LMCS](https://lmcs.episciences.org/18669/pdf) | no code declared |
| P07 | `papers/07_enriched_category_qualia_osf_preprint.docx` | 201,746 | n/a | `ca54d6ff777513534b22b919b2ec18aeb5f62486148d33476047610bb14fdc9d` | [OSF author preprint](https://osf.io/ucjmz/) | no code declared |

P07 is intentionally stored as DOCX. The OSF primary file is named
`22 Mar 26 Enriched Category Paper_formatted_accepted.docx`; the apparent
`/download` endpoint returns this DOCX, not a PDF. Its ZIP structure was tested
successfully, and the SHA256 matches the hash declared by the OSF file API.
No lossy local PDF conversion or unverified third-party copy was substituted.

## Repositories

All repositories are depth-one, no-tag working snapshots. Every working tree was
clean immediately after acquisition.

| ID | Local directory | Role | Remote | Commit | Tree |
|---|---|---|---|---|---|
| R01 | `repos/01_HippoRAG` | official P01 implementation | `https://github.com/OSU-NLP-Group/HippoRAG` | `c617143f01477243992a63b2e2151cc003dd3b21` | `6e35ec10b555e111773c854538035ab217eec1bb` |
| R02a | `repos/02_FQL_archived` | archived predecessor in P02 implementation lineage | `https://github.com/CategoricalData/FQL` | `1ed8ab13de98e74ae3989c7475d1898efac133ae` | `3df79a575fc499c883027778460bc2f4b175c5ef` |
| R02b | `repos/02_CQL_maintained_successor` | maintained P02 reference implementation lineage | `https://github.com/CategoricalData/CQL` | `6beb80a894b4ed1ec328c0bfd4be6b118818c1dc` | `f1124ac3a1237fc8e67d9a6d52ef1e418ab00c17` |
| R04 | `repos/04_KGE-HAKE` | official P04 implementation | `https://github.com/MIRALab-USTC/KGE-HAKE` | `6a82e17855f465d4ec15b880da0d1faaa7c6100c` | `f80d631b195c1f8a9012b31aab56be6a321702b9` |
| R05 | `repos/05_hypergraph_diffusions` | author-linked P05 implementation | `https://github.com/Orecchia-Research-Group/hypergraph_diffusions` | `045e60e5ee3dd72a9ac511c91963f71d789791d7` | `9775db3d2692d6e9d7e6350f997cad505836b6bd` |

The FQL/CQL repositories are an official implementation lineage, not a claim that
either working tree is the exact 2012 experiment snapshot.

The PMLR `mlresearch/v235` repository was not cloned because it is a proceedings
asset store rather than paper-specific code. The Categorical Deep Learning
literature list and website theme were also not misclassified as paper
implementations.

## Current project provenance

- Remote: `git@github.com:erzhu419/Assumption-Agent.git`
- Branch: `codex/reconstruction-v2-paper`
- Commit at acquisition: `ae8eb7f6733be38eab2a1e03d3ffa1f8d175e009`

The current project was not recursively copied into its own reference directory.

## Verification

- Six PDF payloads passed `pdfinfo` and first-page Poppler rendering.
- The rendered contact sheet was visually inspected; all six title pages matched
  their expected papers and contained no login/error pages.
- The OSF DOCX passed `unzip -t`.
- All five repository HEADs matched their pre-download `git ls-remote` values.
- All five repository working trees were clean.
- Bundle size after acquisition: approximately 645 MiB.

