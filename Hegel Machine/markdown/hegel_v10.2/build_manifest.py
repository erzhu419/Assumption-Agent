from pathlib import Path
import json, hashlib
from urllib.parse import quote
root = Path(__file__).resolve().parent
commit='a66cd82b22f444044dbda98f43782c96f833a4f0'
base=f'https://github.com/erzhu419/Assumption-Agent/blob/{commit}/'
entries=[
('S01','上传交接档案','hegel_v10_ai_handoff_and_audit_dossier_20261116.md','全文332行','uploaded dossier / turn401file0, turn405file0',None),
('S02','世界承诺提取','Hegel Machine/src/hegel_machine/cognition/world_structure_commitments.py','1–210','turn388file0','66632c963dc8506e0d71dadea6c2ca852181b533'),
('S03','世界承诺前提/修订','Hegel Machine/src/hegel_machine/cognition/world_structure_commitments.py','460–730;900–1170','turn389file0;turn394file0','66632c963dc8506e0d71dadea6c2ca852181b533'),
('S04','前提图定义','Hegel Machine/src/hegel_machine/cognition/premise_dependency_audit.py','1–230','turn395file0','b64b1a0eae381eeca5a5d9de917401f455ceef4f'),
('S05','前提检验与有限解释','Hegel Machine/src/hegel_machine/cognition/premise_dependency_audit.py','3800–4060','turn397file0','b64b1a0eae381eeca5a5d9de917401f455ceef4f'),
('S06','LLM来源与共享资料','Hegel Machine/scripts/run_v10_phase2b_open_grammar_llm.py','1–260','turn391file0','4bc66d05fc08419b0d0e5b91cce0999fe66977b4'),
('S07','LLM比较与确认写入','Hegel Machine/scripts/run_v10_phase2b_open_grammar_llm.py','305–530','turn392file0','4bc66d05fc08419b0d0e5b91cce0999fe66977b4'),
('S08','LLM作者与允许语法','Hegel Machine/src/hegel_machine/cognition/open_grammar_author.py','1–230','turn393file0','f57f2ccb067e32cf954ea1c86f1a8d8eb1e8dc89'),
('S09','流形对照脚本','Hegel Machine/scripts/run_v10_phase5a_manifold_and_graph_pde.py','1–310','turn387file0','102d015251f43f3618314ec7ff31da5d8e569922'),
('S10','代数核与有限元数','Hegel Machine/src/hegel_machine/cognition/algebraic_proof_certificates.py','1–250;1–90复核','turn380file0;turn406file0','1f9f872a4a4e34e6ea3a93e45a3e06bf3a9369eb'),
('S11','证书验证器与域','Hegel Machine/src/hegel_machine/cognition/algebraic_proof_certificates.py','250–550','turn381file0','1f9f872a4a4e34e6ea3a93e45a3e06bf3a9369eb'),
('S12','Lean导出与不变量合成','Hegel Machine/src/hegel_machine/cognition/algebraic_proof_certificates.py','540–820','turn383file0','1f9f872a4a4e34e6ea3a93e45a3e06bf3a9369eb'),
('S13','实验室/可达求解','Hegel Machine/src/hegel_machine/cognition/algebraic_proof_certificates.py','795–1060','turn384file0','1f9f872a4a4e34e6ea3a93e45a3e06bf3a9369eb'),
('S14','证明基准与硬编码模型','Hegel Machine/scripts/run_v10_phase5c_algebraic_proofs_and_impossibility.py','1–300','turn386file0','3f2ded00d99028cf3a3c88a814b9fef0e19c14c0'),
('S15','前提列表生成','Hegel Machine/src/hegel_machine/cognition/algebraic_proof_certificates.py','1055–1145','turn400file0','1f9f872a4a4e34e6ea3a93e45a3e06bf3a9369eb'),
('S16','单一机制修复器','Hegel Machine/src/hegel_machine/cognition/algebraic_proof_certificates.py','1120–文件末','turn399file0','1f9f872a4a4e34e6ea3a93e45a3e06bf3a9369eb'),
('S17','证明基准原回执','Hegel Machine/docs/evidence/v10_world_structure_20261116/v10_phase5c_algebraic_proofs_and_impossibility_20261116.json','1–230','turn403file0','03fbed6d52030eeb9c87cd8378b50c44ee3d3737'),
('S18','修复基准与名义扫描公式','Hegel Machine/scripts/run_v10_phase5c_algebraic_proofs_and_impossibility.py','300–文件末','turn396file0','3f2ded00d99028cf3a3c88a814b9fef0e19c14c0'),
('S19','COW继承与旧账检查','Hegel Machine/docs/evidence/v10_world_structure_20261116/v10_cow_602_603_new_ledger_20261116.json','1–170，结果另与上传全文对照','turn402file0','22cb5b92bce7303bb74155b17d9e7978f4e11d00'),
('S20','随机环境实际评价','Hegel Machine/scripts/run_v10_phase5b_stochastic_sde_and_resonance.py','1–270','turn404file0','6dde9a39d860c0558f4c966148347859d54ade97'),
('S21','AWM原文','2610.40503v1.pdf','本轮重点复核24/27页；用户全文在前轮已读取','turn405file1; local render page27',None),
('S22','Phase5测试','Hegel Machine/tests/test_v10_phase5_manifold_sde_and_proofs.py','返回内容的前部与证明测试；超长输出后部未全读','turn398file0',None),
('S23','Phase1研究脚本','Hegel Machine/scripts/run_v10_world_structure_study.py','返回内容的前部：COW普查与等价fixture；后部未全读','turn382file0',None),
('S24','v4主线','GPT_diagnosis_v4(3).md','统一语义、支持依赖、双循环；本轮读取原285–384行','turn401file1;turn370file9',None),
('S25','v9修订','hegel_v9.md','用户提供全文关键内容及本对话的先前逐项核查','turn370file7',None),
]
rows=[]
for id,title,path,scope,ref,sha in entries:
    rows.append(dict(id=id,title=title,path=path,scope=scope,retrieval_reference=ref,git_blob_sha1=sha,
                     fixed_url=base+quote(path,safe='/') if path.startswith('Hegel Machine/') else None))
(root/'source_manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
md=['# 来源与核查范围','',f'固定代码版本：`{commit}`。','',
    '代码工具返回的content是JSON字符串，聊天引用通常落在输出L2；下表另提供真实源码路径和所读范围，便于本地逐项复查。超长输出的未见部分不算已读。没有完整checkout，也没有将读取某模块部分范围描述为全模块通过审计。','',
    '| 编号 | 来源 | 实际阅读范围 | 固定位置 |','|---|---|---|---|']
for x in rows:
    loc=f"[{x['path']}]({x['fixed_url']})" if x['fixed_url'] else '`'+x['path']+'`（用户提供）'
    md.append(f"| {x['id']} | {x['title']} | {x['scope']} | {loc} |")
md += ['', '## 实际运行与未运行', '',
       '- 实际运行：`probe_certificate_domain.py`独立Fraction算术；对应结果在JSON中。',
       '- 未运行：仓库验证器本体、13套实验、付费LLM、外部COW盘、Lean编译、全仓测试。',
       '- 没有修改远程仓库、旧回执或任何用户外部工作区。',
       '- 图像：本轮只为检查AWM第27页表格渲染本地PDF；交付包不包含重复论文或字体。',
       '', '## 文档口径', '',
       '本报告所说“源码确认”是特定实现语义；“回执记录”是已存结果；“独立计算”不冒充仓库重跑。文档计数、默认开关和grep命令不是充分的安全/科学有效性证明。']
(root/'03_来源与核查范围.md').write_text('\n'.join(md)+'\n')
findings=[
('F01','P0','代码/证明语义','状态分析域未绑定执行合法性或可达覆盖','S11,S13','独立精确算术反例已运行；仓库直接模式待运行','统一域语义；否则只返回条件不可达'),
('F02','P1','实现与报告','Lean不可达/可达导出缺对应定理，未编译','S12,S17','源码确认','降为draft；需要时实现实际定理及编译'),
('F03','P1','证据归因','主对照同时开关不同求解能力','S09','源码确认','保留能力消融，另做同工具学习对照'),
('F04','P1','证据归因','LLM平铺与门控还改变角色信息/父模型/补丁空间','S06,S07,S08','源码确认','相同模型与工具下隔离依赖因子'),
('F05','P1','报告','四类故障实际是一类端口增益四实例','S16,S18','源码确认','按单一机制报告，不覆盖旧记录'),
('F06','P1','报告/实验','89.92%基线探针来自公式而非真实执行','S18','源码确认','标名义估算；真实匹配执行后才能讲因果收益'),
('F07','P1','实验','rollout认证字段恒false且无同等检查入口','S14,S17','源码与回执一致','共享证人检查器，区分工具对照与架构对照'),
('F08','P1','评价定义','无噪孪生轨迹误差被等同于带噪条件漂移风险','S20','源码确认','重命名并新增同噪声历史条件风险评价'),
('F09','P1','数据身份','独立48行通过后确认元数据仍写validation','S07','源码确认','追加正确确认引用、行数与实际误差'),
('F10','P1','费用','Phase5C求解与检查无SearchBudget参数/扣账','S10,S13,S18','所读路径确认','独立真实费用账；不得称全部计算均已预算'),
('F11','P2','范围','有限元数符号恒等检查被扩大为任意N','S10,S12,S17','源码确认','有限范围表述或补一般证明'),
('F12','P2','证明前提','方向矩正性未完整绑定边方向/非负域','S10,S12','源码分析','检查约束而不只验证恒等式'),
('F13','P2','科学对象','固定前提节点与开发新增家族被写成动态理论发明','S03,S04,S05','源码确认有限性','来源作者分层；保留真实运行时学习'),
('F14','P2','实验范围','Phase5C直接输入完整system_spec而非学得模型链','S14,S16','源码确认','标solver资格验证；实际承诺到证书另验'),
('F15','P2','文档','AR2参数载体文档与实际构造不同','S01,S03','源码与文档对照','由序列化测试生成正确合同示例'),
('F16','P2','证据强度','28项聚焦测试/13JSON并非全仓回归或13独立前瞻研究','S01,S22','档案/测试范围','逐研究标开发/冻结/历史/控制与独立单位'),
('F17','P2','协议解释','基线合法重建相同方法被称为必须修掉的同构病灶','S01,S24','计划与报告对照','允许等价重构并计费，不削弱基线'),
('F18','P2','数学表述','跨离散格点规模不等于连续PDE极限；平移子群不等于全Aut','S01,S09','实现范围/数学辨析','限定为有限离散系统及所测变换'),
('F19','positive','已实现','完整拟合族跨数量实例化和局部前提修订','S02,S03,S04','源码支持','保留并进入同工具生命周期对照'),
('F20','positive','已报告有限收益','COW冷/热v10及零调用迁移显示知识包作用','S01,S07,S19','回执支持，未本轮重跑','保留有限结论并分离来源与目标费用'),
]
data=[dict(id=a,severity=b,kind=c,finding=d,sources=e.split(','),status=f,next_step=g) for a,b,c,d,e,f,g in findings]
(root/'findings.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
md=['# 差距与责任矩阵','', 'P0：可信决策语义须先处理。P1：主要结论或因果归因受影响。P2：范围、维护或表述需收紧。positive：必须保留的实际进展。','',
'| ID | 优先级 | 类型 | 发现 | 核查状态 | 下一步 |','|---|---|---|---|---|---|']
for x in data: md.append(f"| {x['id']} | {x['severity']} | {x['kind']} | {x['finding']} | {x['status']} | {x['next_step']} |")
(root/'04_差距与责任矩阵.md').write_text('\n'.join(md)+'\n')
