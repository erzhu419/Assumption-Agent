from __future__ import annotations
import json
from pathlib import Path
root=Path(__file__).resolve().parent
rows=[
('R01','v4两个循环真正相接','U4 第八/十一节','partial','S,V,D','任务循环有实绩；跨任务理论形成与后继收益未闭合',['G02','G05','G10','G11'],'形成一个实际新对象并改变未参与形成的后继研究','不能把局部拟合改善或方法回放当作跨来源收益','A,I,J'),
('R02','问题/假设族/实例/方法身份分开','U4 第三/四节','implemented_bounded','E_closed','题义版本、查询、程序与方法记录已存在',['G08','G09','G12'],'保留现有合同，只维护真实消费回归','不能通过新建同义类声明又完成一次','B'),
('R03','预测之外的结构约束也是研究对象','U4 第三/六节','partial','S,E,V','有查询与检验；当前归纳主要处理native程序',['G10','G11'],'一条真实关系约束进入候选或实验生成','RelationResidual接口存在不等于已发现关系','A,L'),
('R04','相似性/结构对应/适用性分开','U4 第五节','implemented_bounded','V','有限有序Program和公开映射可比较',['G09'],'新绑定在共同物理语义下接受前瞻检验','当前有限签名相同不能永久合并机制','L'),
('R05','理论之间关系的生成与修订','U4 第六/八节','partial','S,E,V','有限同族、构造及映射；一般关系谱未建立',['G09','G11','G16'],'一个学得关系被后继召回并有实测作用','不用关系标签数或图规模验收','L'),
('R06','所有资料增量处理并形成知识','U4 第八/十一节','partial','E,S,V','材料类别、来源和引用完善；一般内容归纳未闭合',['G12'],'新材料产生可执行对象且不重复计支持','笔记不能提升成独立事实','B,I,L'),
('R07','有答案语义的问题生成','U4 问题程序部分','implemented_bounded','S,V','固定有限查询和确定性触发存在',['G08'],'研究使命下产生具体问题并实际回答/请求作者','缺内容不能用填写query路径伪装解决','B'),
('R08','子研究由父用途触发并返回','U4 第七/九节','partial','E,S,V','绑定/续接存在；部分问题内容仍由协议提供',['G07','G08'],'干预子答案改变真实父输入或实验','只有树节点/文字返回不通过','B,G'),
('R09','并列与局部未知下合法探索','U5 第三节','implemented_executed','E_closed,V','seeded_legal真实购测；局部未知处理已接',['G02','G05'],'维护科学并列与操作选择分离','不声称唯一最优或科学支持提升','F'),
('R10','实际下游表示可诊断','U4 表示；U6 目标充分性','implemented_executed','E_closed,V','闭合native表示暴露并触发15稿研究',['G07','G18','G05'],'新的任务后果检验，不只零碰撞','605个视图不证明正确状态','C,D'),
('R11','无碰撞不阻止合理表示修订','U4 不强求旧语言穷尽','partial_restricted','S,E','program_revision不调用独立coordinates分支',['G07','G18'],'通用策略对持续前瞻失败允许有依据替代路径','保留局部冻结实验，不追溯改搜索','C'),
('R12','问题意义与研究行动价值分开','U4；U5合法探索','partial','S,V','有限G合法但不代表正确解释或后验价值',['G05','G08'],'全部候选错误时研究共同前提而不只换代表','G=1不是增量科学收益','F'),
('R13','学习适用条件与效用范围','U4 scope；U5触发与范围','negative_bounded','S,E,V,D','raw-u半空间对205/260改善标签不可分',['G06'],'分开应用scope、相对效用scope、coverage并检验','not_separable不能排除全部历史条件','E'),
('R14','新父上的再次递归修订','U4递归；U5父锚定','prepared_not_executed','E,R,V','真实新7尝试中断；后续准备未授权',['G02','G20','G21'],'授权后完整反例→新父→修订→比较→冻结读出','兄弟稿胜负不多算递归深度','G'),
('R15','稳定数值与失败边界','U6 D1','completed_bounded_audit','E_closed','723复核16/431/276，零全通过',['G03','G15'],'维护当前数值合同，不重复723','LS实例最大误差超阈不证明整个族不可行','18.6'),
('R16','精确编辑/提案/结构修订分流','U6 D2','implemented_executed','E_closed,S','新20structural_revision且不消费旧精确参数映射',['G02','G16'],'只补当前对象真正依赖的角色与状态','不把旧13输入问题当455失败病因','2,18.5'),
('R17','真实before/after结构归纳','U5第二节','implemented_local_effect','V,D','真实65→267/195来源及新20产生采纳差异',['G02','G16'],'在新来源检验结构和研究收益','更多候选不增加独立来源','I'),
('R18','研究方法/元先验形成','U4理论循环','partial','S,E,V','历史方法学习存在，当前一般归纳与有效性未证',['G14','G11'],'学得方法改变后继生成/研究选择并受检验','宿主写入规则不是系统发现','A,I,M'),
('R19','实际保留研究图的消费','U5复用；U6后继','prepared_not_executed','E_closed,R,V','三依赖真实图接回；4普通+1重放保留',['G07','G20'],'实际训练重放与采纳分开报告','工具调用本身不是用途证明','G'),
('R20','选择/采纳/确认的数据语义','U4证据；U5池分离','partial','E,S,V','9局部holdout胜，7全609胜，已披露',['G02','G05','G07'],'冻结每个选择合同与竞争集合','不把9验证借给7或完整旧池','H'),
('R21','当前对象独立后继收益','U4 PR3；U5/6末阶段','not_established','D,R,V','当前B/C同fixed20，无新增合格跨源对象',['G05'],'有实际不同冻结载荷及未曝光目标','B=C运行不能估计新对象效应','J'),
('R22','问题组织整体增量','U4 ABC；U5强档案','not_established','V,D,R','旧19/20均有问题架构；不识别全组织价值',['G02'],'共同工具历史预算下A/B/C对照','数值/存储修复不能归入PR3处理','J'),
('R23','完整单位预算与授权','U5/6资源边界','prepared_blocked','R,E_closed','2698218计算已报价；新300秒未授权',['G20','G21'],'授权后最多一次完整登记单位并保留停点','不得清零旧账或把评审视为新运行授权','K'),
('R24','知识保留的可计算性','U4有限资源；U5强基线','partial','S,E,V','全池比较占当前报价81.40%，缓存已有进展',['G02','G21'],'新共同合同下严格增量比较或索引','全访问权不等于每个局部步骤从零重算','K'),
('R25','更广关系/尺度/不确定性/多代谱','U4总目标和范围','future_scope','S,V','旧窄域先例不等于统一理论谱完成',['U4','G02','G13','G14'],'首主题闭环后逐项拓展并检验错误迁移','不作为当前第一闭环全部前置','L,M'),
]
items=[]
for id,title,req,status,causes,current,refs,action,forbidden,section in rows:
    items.append(dict(id=id,requirement=title,plan_basis=req,status=status,cause_classes=causes.split(','),current_evidence=current,source_ids=refs,next_behavior=action,invalid_shortcut=forbidden,report_sections=section.split(',')))
payload={
 'audit_commit':'d77b2d7d168187220fe6c850f75c5d59ed3a0111','date':'2026-10-26',
 'status_is_not_completion_percentage':True,
 'authority_order':['user_uploaded_v4_master_goals','v5_local_amendments','v6_target_and_numeric_amendments','current_evidence_for_current_facts'],
 'cause_classes':{'E':'execution/integration','E_closed':'previous execution defect repaired','S':'plan/algorithm limitation','D':'data limitation','R':'resource/authorization','V':'effect/attribution unestablished'},
 'requirements':items,
 'verification_boundary':'Read-only source/evidence audit, arithmetic rechecks only; no full repository tests, scientific fitting, model calls, or new observations.'
}
(root/'requirements_matrix.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
