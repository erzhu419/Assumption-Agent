# Frozen v1/v2 snapshots

这里的 52 个 Python 文件是 v3 启动时实际参考的最小迁移集合：

- `v1_assumption_os/`：framework object、branch/lifecycle、conservative gate、
  formal/simulator routing 与对应测试；
- `v2_gscl/`：UAO/meta-assumption、GSCL schema/exact residual/evidence extractor、
  controlled corpus 与对应测试。

它们不是可独立安装的 vendor package，也不进入 `src/hegel_machine/` 的 import
path。活动实现会重写合同、重新测证据，不能直接继承旧 PASS。

`source_manifest.tsv` 给出：

```text
snapshot_path | original_path | original_git_state | sha256
```

复制时有 40 个源文件已被原仓库跟踪，12 个 GSCL 源码/测试仍是原工作树中的
untracked 文件。这里显式保存它们的内容和 hash，避免以后从 `HEAD` 重建时
悄然丢失。

`source_git_head.txt` 是复制时顶层仓库的 HEAD；它只标识基线，不表示工作树
干净。用户原有的其他修改和未跟踪产物没有进入本目录。
