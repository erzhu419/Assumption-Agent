# Hegel Machine 文献与源码归档

本目录是 `黑格尔机.md` 与 `黑格尔机和泛函分析.md` 的可审计资料基线。它覆盖两份文档中的全部显式外链，也补入了全文中实际承担论证作用、但没有进入末尾链接表的命名文献：Hegel《哲学史讲演》、QuAC、HippoRAG、Lean 4、Mathlib、Co-Scientist 与 Robin。

归档日期：2026-07-30（Asia/Shanghai）。

## 从哪里开始

- `manifest.json`：逐篇文献、逐个仓库的来源、状态、访问限制、固定 commit、文件大小与 SHA-256。
- `references.bib`：可直接导入 Zotero、JabRef 或其他参考文献管理器的 BibTeX。
- `checksums.sha256`：整个目录除其自身之外的平面校验账本。
- `papers/`：26 个经 `pdfinfo` 验证的真实 PDF；访问页和挑战页绝不以 `.pdf` 命名。
- `webpages/`：网页快照、出版商访问页与挑战响应。
- `metadata/`：arXiv Atom、Crossref JSON 与 DOI BibTeX 元数据。
- `repositories/archives/`：19 个固定 commit 的 GitHub `tar.gz` 源码快照，不含 `.git`；DreamCoder/LAPS 的三个 gitlink submodule 也按父仓库记录的精确 commit 单独归档。
- `repositories/metadata/`：GitHub repository/commit API 响应，以及匿名仓库的公开列表和 HTTP 401 响应。

运行下面两条命令可复核归档：

```bash
sha256sum -c checksums.sha256
python3 tools/update_integrity.py
```

第二条命令会重算 `manifest.json` 中的大小和哈希，并刷新 `checksums.sha256`。源码快照可用 `tar -xzf <archive>.tar.gz` 解开；它们是可复现的源码树快照，不是带历史的 Git clone。

## 归档口径

“论文里的 repo”按以下口径处理：

1. 论文或补充材料明确给出的代码、数据、实验结果仓库；
2. 被正文点名且直接承担实现职责的基础设施仓库；
3. 后续作者关联的 companion repository，但必须标明它不是原系统引擎；
4. 不镜像仅出现在参考文献中的下游仓库、通用依赖、出版站点仓库和第三方复现。

因此，JAX、XLA、AI-Researcher、PMLR 站点仓库及数十个 bibliography-only URL 没有被错误地包装成论文 artifact；其处置原因在 `manifest.json` 的 `repository_dispositions` 中。

## 已固定的关键仓库

| 资料 | 性质 | 固定状态 |
|---|---|---|
| `SakanaAI/AI-Scientist-v2` | 论文官方代码 | 固定 commit |
| `SakanaAI/AI-Scientist-ICLR2025-Workshop-Experiment` | 论文官方实验 artifact | 固定 commit |
| `google-deepmind/alphaevolve_results` | 论文直接链接的结果 notebook；不是 AlphaEvolve 引擎 | 固定 commit |
| `google-deepmind/alphaevolve_repository_of_problems` | 后续作者关联 companion；明确不含 AlphaEvolve 引擎 | 固定 commit |
| `ellisk42/ec` 默认分支及 `icml_2021_supplement` | DreamCoder 主仓库与 LAPS 补充材料短链解析到的精确分支；三个 submodule 也按 gitlink 固定 | 两个父 commit + 三个 gitlink commit |
| `bvarici/score-general-id-CRL` | 论文发表前状态和 legacy 仓库 HEAD 各一份 | 两个固定 commit |
| `acarturk-e/score-based-crl` | legacy README 指向的现行维护仓库 | 固定 commit |
| `ellisk42/humanlike_fewshot_learning` | hypothesis-generation 论文使用的公开数据 | 固定 commit |
| `OSU-NLP-Group/HippoRAG` | 论文官方代码 | 固定 commit |
| `Future-House/robin`、`Future-House/finch` | Robin 与其数据分析组件 | 各固定 commit |
| `leanprover/lean4`、`leanprover-community/mathlib4` | 文中点名的验证基础设施 | 各固定 commit |
| `openai/codex` | OpenAI agent-loop 文章直接关联的实现 | 固定 commit |

每个完整 commit、分支名、API 元数据文件和压缩包哈希均以 `manifest.json` 为准。

## 未伪装成成功的项目

| 项目 | 实际结果 |
|---|---|
| OpenAI 两篇 Codex 网页 | 直接请求获得挑战页，不是正文；挑战响应被明确命名并保留 |
| Torgersen, *Deficiencies* | Cambridge 仅返回 landing/access 页面；章节全文受限 |
| Huber, *Belief Revision I* | Wiley PDF/HTML 均返回 HTTP 403；保留 Crossref 元数据与 403 响应 |
| 匿名 `number_game-F153` | 文件列表公开，但整库下载端点返回 HTTP 401 `not_connected`；未登录、未递归抓取规避 |
| AlphaEvolve | 官方引擎没有公开；这里只归档 paper-linked results 和后续 companion |
| RQGM、ADVENT、Co-Scientist | 未发现作者公开的系统实现仓库；没有用同名第三方实现冒充 |

Hegel 卷一采用 Project Gutenberg 的公版 Haldane 英译本。项目讨论中的中文表述应视为思想概括；本归档不声称它是逐字原话。

泛函分析部分另补入 Hilbert 1912 年《线性积分方程的一般理论纲要》的公版扫描，标记为 `manual_derived_historical_source`。Hilbert 在序言中说明该卷重印 1904–1910 年间的六篇通信，因此不能把文档中“1906 单篇”的简写当作精确书目事实。

## 完整性说明

所有 `papers/*.pdf` 都已通过 `pdfinfo`；所有 `repositories/archives/*.tar.gz` 都已通过 `gzip -t`。`manifest.json` 是机器可读的权威清单，README 只提供导航，不替代其中的逐项证据。
