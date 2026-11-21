"""Build a complete auditable index of all v18.4 failing in-scope episodes."""
from pathlib import Path
import os
import json
from urllib.parse import quote

OUT=Path(__file__).resolve().parent
ROOT=(Path(os.environ.get('HEGEL_V18_REPO', str(Path(__file__).resolve().parent.parent / 'repo_v18_922b95e19'))) / 'Hegel Machine')
SHA='922b95e194d17c9259f1993706440e0602a43db9'
data=json.loads((OUT/'v18_recomputed.json').read_text())
replay_path=OUT/'v18_replayed_runtime_errors.json'
replay=json.loads(replay_path.read_text())['results'] if replay_path.exists() else []
replayed={x['code_path']:x for x in replay}
cats={'false_confirmation':'确认已过，但最终 heldout 失败',
      'provider_errors':'未提交候选的 provider/环境故障',
      'execution_errors':'轨迹有 Python 执行错误（不自动判为唯一原因）',
      'confirmation_refuted':'作者程序被确认集拒绝',
      'direct_abstain':'未提交候选即 abstain，算法采纳未证实'}
arms={'Hegel_M5_LiveGPT':'Hegel','EqualAuth_Flat_LiveGPT':'Flat','Strong_Ordinary_LiveGPT':'Strong'}
def link(path,label):
    rel='Hegel Machine/'+str(path.relative_to(ROOT))
    return f'[{label}](https://github.com/erzhu419/Assumption-Agent/blob/{SHA}/{quote(rel)})'
out=['# v18.4 全部 104 个域内失分逐题索引',
     '',f'固定提交 `{SHA}`。主诊断见 `v18_diagnosis.md`；完整可复算字段见 `v18_recomputed.json`。',
     '', '本附件逐一保留原始结果、原动作顺序、确认记录与作者代码路径。分类是轨迹观察，不把所有出现 execution error 的 episode 都宣称为单一接口原因，也不把没有作者提交的 provider 失败叫作科学求解失败。可复现接口/符号域案例另在 targeted controls 中；它们没有改写下面任何原成绩。',
     '', '本地 runtime 重放保留了原源码 SHA 和完整本轮 traceback，见 v18_replayed_runtime_errors.json。它不是原机器 stderr 的逐字恢复：本轮 timeout=15s，原 host timeout=25s，环境也有差别；四个本地 timeout 仅表示此诊断上限内未完成。',
     '', '## 分类索引','', '| 模型 | 臂 | 题 | 观察分类 | confirm RMSE | heldout RMSE |','|---|---|---|---|---:|---:|']
cases=sorted(data['failure_cases'],key=lambda r:(r['model'],r['arm'],r['continual_wave'],r['task_id']))
for i,r in enumerate(cases,1):
    out.append(f"| {r['model']} | {arms[r['arm']]} | {i:03d}: `{r['task_id']}` | {cats[r['failure_bucket']]} | {r['confirm_rmse']} | {r['heldout_rmse']} |")
for i,r in enumerate(cases,1):
    ws=ROOT/r['relative_workspace']
    out += ['',f"## {i:03d}. {r['model']} · {arms[r['arm']]} · {r['task_id']}",'',
            f"**原结论**：`{r['status']}`；strict_pass={r['strict_pass']}；false_confirmation={r['false_confirmation']}；候选数={r['evaluated_candidate_count']}；调用数={r['api_calls']}；C={r['composite_cost']}。",'',
            f"**观察分类**：{cats[r['failure_bucket']]}。",'',
            link(ws/'decision_provenance_dag.json','原始完整 DAG 与终局摘要')]
    if r['decisions']:
        out += ['','### 作者工具决策','']
        for t in r['decisions']:
            why=str(t.get('rationale','')).replace('\n',' ')
            out.append(f"- Turn {t.get('turn')}: `{t.get('action')}` — {why}")
    if r['confirmations']:
        out += ['','### 实际执行与确认','', '| 候选 | train RMSE | confirm RMSE | tolerance |', '|---|---:|---:|---:|']
        for c in r['confirmations']:
            out.append(f"| {str(c.get('hypothesis_name','')).replace('|','/')} | {c.get('train_rmse')} | {c.get('confirm_rmse')} | {c.get('tolerance')} |")
    if r['errors']:
        out += ['','### 记录中的错误','']
        # Compress identical infrastructure errors, preserving their occurrence count.
        counts={}
        for e in r['errors']:
            s=json.dumps(e,ensure_ascii=False) if not isinstance(e,str) else e
            counts[s]=counts.get(s,0)+1
        for s,n in counts.items():
            out += [f'- {n} 次：`{s.replace(chr(10), " ").replace("`", "'")}`']
    models=sorted(ws.glob('model_step_*.py'))
    if models:
        out += ['','### 保存的作者程序','']
        for p in models:
            out.append('- '+link(p,p.name))
            rr=replayed.get(str(p))
            if rr:
                out.append('  本轮只读本地错误复现：'+('`'+str(rr.get('exception_type',''))+': '+str(rr.get('error','')).replace('`',"'")+'`' if rr.get('exception_type') else json.dumps({k:v for k,v in rr.items() if k not in {'model','arm','task_id','code_path','payload_path','traceback'}},ensure_ascii=False)))
    else:
        out += ['','没有提交可执行候选文件；因此此记录本身不能用于推断科学模型拟合能力。']
(OUT/'v18_failure_appendix.md').write_text('\n'.join(out)+'\n')
print(f'{len(cases)} failing episodes; {len(replay)} local runtime replays; {len(out)} lines')
