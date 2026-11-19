# 当前分支文件覆盖计数复核

日期：2026-11-19（Asia/Shanghai；UTC 2026-11-18）。仓库：`erzhu419/Assumption-Agent`。本次使用已取得的分支根与完整递归子树数据重建索引，没有重复下载既有文件或历史提交；只补取此前未进入索引的一份 Markdown 原文。

## 1. 结论与正确计数

初始文件索引确实漏纳了后来补取的两个 `reference` 根子树。初始清单包含47个不同根子树，完整清单包含49个；49个均标记为非截断，无错误。16个分支根引用的所有子树均已在完整清单中找到，分支名称和固定 head 与 `branches.json` 一致。

| 统计口径 | 修正前 | 修正后 |
|---|---:|---:|
| 固定分支 head | 16 | 16 |
| 不同根子树 | 47 | 49 |
| 全文件路径版本 `(path, blob SHA)` | 11,958 | **11,960** |
| 全文件不同路径 | 11,856 | **11,858** |
| 全文件不同内容 blob | 11,247 | **11,249** |
| Markdown 路径版本 | 1,811 | **1,812** |
| Markdown 不同路径 | 1,802 | **1,803** |
| Markdown 不同内容 blob | 1,516 | **1,517** |

这里的全文件统计是目录/内容身份清单，不表示所有 PDF、数据及源码均已下载全文。原先的 `11247` 是不同内容 blob 数，不是不同路径数。共享内容可在不同路径和多个分支重复出现，所以这些口径不能混用。

遗漏的两行是：

| 路径 | blob SHA | 所在分支 |
|---|---|---|
| `reference/Automated Hypothesis Validation with Agentic Sequential Falsification.pdf` | `080065a912e760b2c32b9d5db25689aa0b480ff3` | 认知主线 |
| `reference/repos_manifest.md` | `5d139882b3a402228f2ae65d5ee7a3752c912709` | 全部16个固定分支 |

两个新增 `reference` tree 分别为 `e5a72343c0da66495d2509b426e59c6d2bef2ad1` 和 `118ae70cf7a6e3d817de298a9242fd61fb3fd8cc`。它们共享同一份 `repos_manifest.md`；认知主线另有PDF。因此不是新增三个文件对象，而是新增两个。

重建使用每个分支根的真实 `(prefix, tree SHA)` 展开递归条目，避免把同一子树的多个路径、分支列表进行不准确的笛卡尔积。既有11,958行的 branch/ref 关联与重建结果相同，没有需要修改的旧关联，也没有多余旧行。

## 2. Markdown 原文完整性

此前取得的1,516个 blob 对原有 Markdown 清单是完整的，但原清单本身遗漏了 `reference/repos_manifest.md`，所以不能继续称它覆盖全部当前分支 Markdown。

该文件本次按固定认知主线 head `787256bd415515905359626f6a5498174ad5db3c` 补取，大小3,581字节，写盘前后均按 Git blob 哈希验证。随后重新校验全部对象：

- **1,517 / 1,517** 个所需 Markdown blob 已在本地。
- 总字节数 **30,840,173**。
- Git blob SHA1 不匹配：**0**。
- 缺失 Markdown blob：**0**。
- 索引外 Markdown blob：**0**。
- 本代理累计成功抓取 **1,106** 个，失败 **0**；其中1个为本次覆盖修正后补取。

此前400个文件各有一个额外尾换行的问题已独立解决：仅在去掉该字节后哈希恰好等于目标 SHA 时改写，明细在 `md_integrity_repairs_scope.json`。此次复核没有再次调整原文内容。

Markdown 范围按文件路径后缀 `.md`（忽略大小写）定义；此处的完整性针对这16个固定分支 head 的当前树，不扩展为1627个历史提交中所有中间、删除或不可达版本的全文覆盖。

## 3. 历史指标检索正则复核

实际读取 `search_corpus.py` 的 AST 字符串，并以 Python 正则和真实 `rg` 分别控制验证。规则为：

```regex
72pp|98\.9|recall.{0,16}0\.0|v16_final_results|formalization_v2_report
```

两个小数点前各有**一个**反斜线，总计两个反斜线字符。不存在连续双反斜线。JSON或嵌套工具输出会为反斜线再转义，不能据显示中的双写判定源字符串错误。

| 控制输入 | 预期 | Python | rg |
|---|---|---|---|
| `98.9` | 命中 | 命中 | 命中 |
| `98x9` | 不命中 | 不命中 | 不命中 |
| `recall=0.0` | 命中 | 命中 | 命中 |
| `recall=0x0` | 不命中 | 不命中 | 不命中 |

已保存全文索引中的规则与脚本一致。因此**本次没有发现这条正则的转义错误**，没有证据声称此前检索漏掉这些小数表达。

修正 Markdown 清单后，已重跑所有七组全文检索。历史指标主题仍命中10个内容对象、19行。完整结果在 `fulltext_topic_index.json`；新的 `fulltext_search_notes.md` 自动引用实际索引计数，去除了原先硬编码的1811/1516。全文检索不等于逐行理解，关键词未命中也不能证明概念不存在。

## 4. 已更新文件与复核边界

已更新：

- `all_branch_file_versions.json`、`markdown_versions.json`：从49个完整子树重建。
- `md_fetch_queue.json`、`md_fetch_progress_scope.json`：加入遗漏对象及准确数量。
- `md_fetch_final_verification_scope.json`：全部1,517个内容对象的最终哈希/范围核对。
- `history_coverage_summary.json`、`history_coverage.md`：仅修正当前文件索引数量；1627个历史提交元数据与父链结论不变。
- `search_corpus.py`：全文覆盖数量从输入动态取得；检索表达式未改动。
- `fulltext_topic_index.json`、`fulltext_search_notes.md`、`initial_thought_paths.json`：按新清单重生成。

统计明细在 `coverage_reconciliation.json`，本地重建过程在 `reconcile_coverage.py`。脚本不调用网络；重新执行时，“修正前”会反映执行当时的输入，所以原始遗漏事实以此次已保存统计和本记录为准。

本次没有修改项目仓库或主诊断报告，没有重跑科学实验，没有把目录覆盖等同于代码正确性或实验效用证明。
