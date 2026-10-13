# 实验数据归档（2026-10-12）

807份历史大JSON、48份近期重复状态文件、旧repair作者的trials和repair_archive已完整压缩到 `/mnt/c/Users/erzhu/Hegel-History-Archive-20261012/`，解压逐字节比对通过后移除原件。另清掉未进入模型调用的continued恢复复制品。累计移出59.23GB原始内容，归档占C盘4.08GB；近期部分实际释放13.27GB。

最新recurrent_experiment_design最终结果及36次作者实践、prospective失败作者及37次实践保留，可继续分析和续跑。代码、论文、摘要、失败诊断和fixture保留。清掉的active.json均有对应最终training.json。

需重算旧明细或恢复旧repair作者时，先按归档目录的README.md和original_paths.json恢复对应gzip文件或tar.gz子目录；不要重新跑已有实验。当前续跑不依赖这些已归档原件。中断恢复目录内的RUN_STATUS.md记录了可重建复制品及续跑来源。

## 2026-10-13 D盘旧实验归档完成

732个单元的91.75GB当前剩余旧内容已完整归档到 `/mnt/d/Hegel-History-Archive-20261012/`，tar.gz共27.36GB。每项完整解压逐字节比对、核对目录清单后，移除48,654个计划中的大文件；WSL artifacts实占136.60→54.73GB，实际减少81.86GB。原地保留Git文件、源码、图片、小型摘要/协议、模型执行上下文及恢复指针；近期实验和PID421601未动，0项归档失败、0个Git文件缺失。

恢复方法见D盘README及各旧目录的ARCHIVE_POINTER.json；只需恢复对应实验，无须重跑付费调用。更早归档在C盘的资料仍需保留，本批没有迁移或删除它们。WSL虚拟磁盘的宿主空间由用户之后压缩回收。[本批结果](storage_inventory_20261012/cleanup_result.json)。
