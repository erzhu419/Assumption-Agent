# v0.45 实验测试
保存 8 项纯噪声测试与 4 项 SC-KG 集成测试；正式默认算法已恢复，实验未采用。
从项目根目录执行以下命令：先把冻结源码加入 cognition 搜索路径，再加入原 tests 路径以导入既有 fixture。
```bash
PYTHONPATH=src python3 -c 'import sys,pathlib,pytest; import hegel_machine.cognition as c; r=pathlib.Path.cwd(); e=r/"artifacts/sc_kg_noise_revision_v045_20260910"; c.__path__.insert(0,str(e/"method_source_v045")); sys.path.insert(0,str(r/"tests")); raise SystemExit(pytest.main(["-q",str(e/"experiment_tests")]))'
```
原通过日志：8 项纯测试包含于 [33 项核心测试](../../noise_revision_v045_final_core_tests_20260910.log)；另见 [4 项集成测试](../../noise_revision_integration_v045_contrasts_20260910.log) 与 [46 项控制器测试](../../noise_revision_v045_controller_tests_20260910.log)。
