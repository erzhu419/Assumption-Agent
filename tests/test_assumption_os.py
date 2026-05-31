import json
import tempfile
import unittest
from pathlib import Path

from assumption_os.adapters import ingest_artifacts, load_exp82_hypotheses, load_wisdom_nodes
from assumption_os.activation import build_activation_profile
from assumption_os.candidate_acceptance import AcceptanceDecision, apply_accepted_candidates, build_acceptance_payload
from assumption_os.conditioned_eval import (
    ConditionedEvalRow,
    GateDecision,
    GateThresholds,
    RouteLabel,
    evaluate_node,
    route_problem_to_node,
)
from assumption_os.domain_templates import format_phase2_domain_execution_template
from assumption_os.graph_memory import JsonlGraphStore, SimpleAssumptionGraph
from assumption_os.lifecycle import LifecycleActionType, plan_lifecycle_actions
from assumption_os.candidate_eval import CandidateReadiness, build_candidate_eval_payload
from assumption_os.proposal_overlay import apply_proposal_overlay
from assumption_os.proposals import ProposalType, build_candidate_proposals
from assumption_os.record_phase2_eval import record_phase2_eval
from assumption_os.retrieval_policy import retrieve_phase2_assumptions
from assumption_os.residuals import classify_manifest
from assumption_os.schema import (
    AssumptionEdge,
    AssumptionNode,
    AssumptionType,
    EdgeType,
    ResidualType,
    TrialManifest,
    TrialStatus,
)
from assumption_os.selector import MetaproductivitySelector


class AssumptionOSTest(unittest.TestCase):
    def test_schema_round_trip_and_retrieval(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            a = AssumptionNode(
                id="strategy_S15",
                type=AssumptionType.METHOD,
                claim="从最小可工作版本开始，逐步添加功能",
                context_conditions=["复杂系统", "高风险"],
                tags=["incremental", "增量构建", "S15"],
                confidence=0.8,
                metaproductivity=0.3,
            )
            b = AssumptionNode(
                id="strategy_S01",
                type=AssumptionType.METHOD,
                claim="固定其他条件，每次只改变一个因素",
                tags=["控制变量", "S01"],
                confidence=0.75,
            )
            store.upsert_node(a)
            store.upsert_node(b)
            store.add_edge(AssumptionEdge(source="strategy_S15", target="strategy_S01", type=EdgeType.DEPENDS_ON))
            store.flush()

            graph = SimpleAssumptionGraph(JsonlGraphStore(td))
            activated = graph.retrieve("世界模型外推失败，应该先做最小场景并替换一个核心模块", seeds=["S15"], top_k=2)
            self.assertEqual(activated.nodes[0].id, "strategy_S15")
            self.assertIn("strategy_S01", {n.id for n in activated.nodes})

    def test_retrieval_can_filter_primary_assumption_types(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            store.upsert_node(AssumptionNode(
                id="strategy_S01",
                type=AssumptionType.METHOD,
                claim="固定其他条件，每次只改变一个因素",
                tags=["控制变量", "S01"],
                confidence=0.75,
            ))
            store.upsert_node(AssumptionNode(
                id="case_1",
                type=AssumptionType.CASE,
                claim="一次营销实验案例反复提到控制变量和小额测试",
                tags=["case", "S01"],
                confidence=0.9,
            ))
            store.flush()

            graph = SimpleAssumptionGraph(JsonlGraphStore(td))
            activated = graph.retrieve(
                "控制变量 小额测试",
                top_k=2,
                candidate_types={AssumptionType.METHOD},
            )
            self.assertEqual([n.id for n in activated.nodes], ["strategy_S01"])

    def test_trial_update_keeps_execution_lapse_from_penalizing_assumption(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            store.upsert_node(AssumptionNode(id="a1", type=AssumptionType.METHOD, claim="use controlled variables", confidence=0.7))
            graph = SimpleAssumptionGraph(store)
            trial = TrialManifest(
                problem_id="p1",
                action_type="strategy",
                assumption="control one variable",
                why_selected="high coupling risk",
                expected_effect="localize failure",
                assumption_ids=["a1"],
                residual="The plan was valid but not applied in the answer.",
                residual_type=ResidualType.EXECUTION_LAPSE,
                status=TrialStatus.FAILED,
            )
            graph.update_from_trial(trial, persist=False)
            self.assertAlmostEqual(store.nodes["a1"].confidence, 0.7)
            self.assertTrue(store.nodes["a1"].residual_ids)

    def test_residual_classifier(self):
        trial = TrialManifest(
            problem_id="p1",
            action_type="audit",
            assumption="selected wisdom should shape answer",
            why_selected="selection score high",
            expected_effect="answer uses wisdom",
            residual="草稿只是表面提及 wisdom，没真正执行",
        )
        assessed = classify_manifest(trial)
        self.assertEqual(assessed.residual_type, ResidualType.EXECUTION_LAPSE)

    def test_wisdom_and_exp82_adapters(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wisdom_path = root / "wisdom.json"
            wisdom_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "W001",
                            "aphorism": "先立后破",
                            "source": "民间谚语",
                            "signal": "要改系统但风险高时",
                            "unpacked_for_llm": "先保留已验证部分，只替换一个新模块。",
                            "cross_domain_examples": [{"domain": "software", "scenario": "替换核心模块前保留旧管线。"}],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            hypo_path = root / "hypotheses.jsonl"
            hypo_path.write_text(
                json.dumps(
                    {
                        "hid": "abc",
                        "seed_cid": "WCAND01",
                        "kind": "decomposition",
                        "claim": "split problem into verifyable stages",
                        "expr": {"steps": ["find assumptions", "verify"]},
                        "trigger_subset": ["p1"],
                        "outside_subset": ["p2"],
                        "evidence": {"delta_ext_base": 0.12},
                        "decision": "accepted",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            store = JsonlGraphStore(root / "graph")
            ingest_artifacts(store, [load_wisdom_nodes(wisdom_path), load_exp82_hypotheses(hypo_path)])
            self.assertIn("wisdom_W001", store.nodes)
            self.assertIn("hyp_abc", store.nodes)
            self.assertTrue(store.evidence)
            ranked = MetaproductivitySelector(SimpleAssumptionGraph(store)).rank("verifyable stages", top_k=2)
            self.assertTrue(ranked)

    def test_record_phase2_eval_writes_trials_and_residuals(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            graph_dir = root / "graph"
            store = JsonlGraphStore(graph_dir)
            store.upsert_node(AssumptionNode(
                id="strategy_S01",
                type=AssumptionType.METHOD,
                claim="固定其他条件，每次只改变一个因素",
                tags=["S01", "控制变量"],
                confidence=0.6,
            ))
            store.flush()

            sample_path = root / "sample.json"
            sample_path.write_text(json.dumps([
                {
                    "problem_id": "p1",
                    "domain": "business",
                    "difficulty": "medium",
                    "description": "预算有限时先小额测试不同渠道。",
                    "coverage_tags": ["S01"],
                }
            ], ensure_ascii=False), encoding="utf-8")
            meta_path = root / "meta.json"
            meta_path.write_text(json.dumps({
                "p1": {
                    "frame": "hybrid",
                    "critical_reframe": "用小实验定位有效渠道。",
                    "rewritten_problem": "设计小额对照实验。",
                    "what_changed": "显式化预算约束。",
                    "anti_patterns": [],
                }
            }, ensure_ascii=False), encoding="utf-8")
            judgment_path = root / "judgments.json"
            judgment_path.write_text(json.dumps({
                "p1": {
                    "winner": "baseline",
                    "score_a": 8,
                    "score_b": 9,
                    "reasoning": "baseline 更完整。",
                    "a_was": "A",
                }
            }, ensure_ascii=False), encoding="utf-8")

            summary = record_phase2_eval(
                root=root,
                graph_dir=graph_dir,
                sample_path=sample_path,
                meta_path=meta_path,
                judgment_paths=[judgment_path],
                intervention_variant="ag",
                baseline_variant="baseline",
                eval_id="unit_eval",
                top_k=1,
            )
            updated = JsonlGraphStore(graph_dir)
            self.assertEqual(summary["outcomes"], {"loss": 1})
            self.assertEqual(summary["residual_types"], {"optimization": 1})
            self.assertEqual(len(updated.trials), 1)
            self.assertTrue(updated.evidence)
            self.assertTrue(updated.nodes["strategy_S01"].residual_ids)

    def test_software_engineering_reranker_boosts_execution_specific_methods(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            for sid, claim in [
                ("strategy_S22", "重新定义问题本身"),
                ("strategy_S24", "识别关键瓶颈节点"),
                ("strategy_S12", "用新证据更新判断"),
                ("strategy_S11", "设定足够好的发布阈值"),
                ("strategy_S08", "提出猜测并测试"),
            ]:
                store.upsert_node(AssumptionNode(
                    id=sid,
                    type=AssumptionType.METHOD,
                    claim=claim,
                    tags=[sid.replace("strategy_", "")],
                    confidence=0.7,
                ))
            store.flush()
            graph = SimpleAssumptionGraph(JsonlGraphStore(td))
            result = retrieve_phase2_assumptions(
                graph,
                problem="QA Lead needs to prioritize many release-blocking bugs before launch.",
                meta={"frame": "paradigm", "critical_reframe": "按玩家影响、收入、回归风险制定发布前修复阈值"},
                pid="se_bug",
                domain="software_engineering",
                difficulty="medium",
                top_k=3,
                pool_k=5,
                skip_domains=set(),
            )
            self.assertEqual(result.diagnostics["route"], "release_quality")
            ranked = [n.id for n in result.subgraph.nodes]
            self.assertIn("strategy_S24", ranked)
            self.assertIn("strategy_S12", ranked)
            self.assertIn("strategy_S11", ranked)
            self.assertTrue(result.policy_notes)

    def test_software_engineering_template_is_route_specific(self):
        release_template = format_phase2_domain_execution_template(
            "software_engineering",
            "QA Lead needs to prioritize many release-blocking bugs before launch.",
            {"critical_reframe": "按玩家影响、收入、回归风险制定发布前修复阈值"},
        )
        self.assertIn("release_quality", release_template)
        self.assertIn("release gate", release_template)
        self.assertIn("rollback/kill-switch", release_template)

        adapter_template = format_phase2_domain_execution_template(
            "software_engineering",
            "Discover an undocumented device API and build an adapter MVP safely.",
            {},
        )
        self.assertIn("adapter_discovery", adapter_template)
        self.assertIn("capability matrix", adapter_template)

        self.assertEqual(format_phase2_domain_execution_template("business", "渠道预算怎么分配", {}), "")

    def test_conditioned_evaluator_routes_and_gates_by_relevance(self):
        node = AssumptionNode(
            id="strategy_S01",
            type=AssumptionType.METHOD,
            claim="固定其他条件，每次只改变一个因素",
            tags=["S01", "控制变量"],
            payload={"activation": {"domains": ["software_engineering"]}},
        )
        rows = [
            ConditionedEvalRow(
                problem_id="p1",
                domain="software_engineering",
                difficulty="medium",
                description="线上线下指标不一致，需要控制变量排查。",
                coverage_tags=["S01"],
                outcome="win",
                active_assumption_ids=["strategy_S01"],
            ),
            ConditionedEvalRow(
                problem_id="p2",
                domain="software_engineering",
                difficulty="medium",
                description="定位性能回退，需要一次只改一个因素。",
                coverage_tags=["S01"],
                outcome="win",
                active_assumption_ids=["strategy_S01"],
            ),
            ConditionedEvalRow(
                problem_id="p3",
                domain="business",
                difficulty="medium",
                description="渠道预算分配。",
                coverage_tags=["S21"],
                outcome="loss",
                active_assumption_ids=["strategy_S01"],
            ),
        ]

        self.assertEqual(route_problem_to_node(node, rows[0]), RouteLabel.SHOULD_FIRE)
        self.assertEqual(route_problem_to_node(node, rows[2]), RouteLabel.NO_FIRE)
        summary = evaluate_node(node, rows, thresholds=GateThresholds(min_benefit_n=2, min_harm_n=1))
        self.assertEqual(summary.decision, GateDecision.NARROW_SCOPE)
        self.assertEqual(summary.active_should_fire_outcomes, {"win": 2})
        self.assertEqual(summary.active_no_fire_outcomes, {"loss": 1})

        rows[2] = ConditionedEvalRow(
            problem_id="p3",
            domain="business",
            difficulty="medium",
            description="渠道预算分配。",
            coverage_tags=["S21"],
            outcome="win",
            active_assumption_ids=[],
        )
        summary = evaluate_node(node, rows, thresholds=GateThresholds(min_benefit_n=2, min_harm_n=1))
        self.assertIn(summary.decision, {GateDecision.KEEP, GateDecision.PROMOTE})

    def test_conditioned_strategy_routing_does_not_fall_back_to_broad_lexical_match(self):
        node = AssumptionNode(
            id="strategy_S15",
            type=AssumptionType.METHOD,
            claim="从最小可工作版本开始，逐步添加功能，通过迭代循环不断完善和扩展产品或系统。",
            tags=["S15", "incremental"],
        )
        unrelated = ConditionedEvalRow(
            problem_id="p1",
            domain="software_engineering",
            difficulty="medium",
            description="评估医疗设备的商业化路径和责任边界。",
            coverage_tags=["S21", "S23"],
            outcome="win",
            active_assumption_ids=[],
        )
        relevant = ConditionedEvalRow(
            problem_id="p2",
            domain="software_engineering",
            difficulty="hard",
            description="给遗留系统设计最小可行增量替换路径。",
            coverage_tags=["S15"],
            outcome="win",
            active_assumption_ids=["strategy_S15"],
        )

        self.assertEqual(route_problem_to_node(node, unrelated), RouteLabel.NEUTRAL)
        self.assertEqual(route_problem_to_node(node, relevant), RouteLabel.SHOULD_FIRE)

    def test_wisdom_routing_uses_trigger_profile_not_broad_lexical_match(self):
        wisdom = AssumptionNode(
            id="wisdom_W020",
            type=AssumptionType.METHOD,
            claim="当你犹豫是继续投入还是及时退出时，区分责任感、脸面、沉没代价和不甘心。",
            tags=["wisdom", "W020"],
            context_conditions=["当你在继续与撤回之间摇摆，既受惯性牵引又怕显得软弱时。"],
            payload={
                "signal": "当你在继续与撤回之间摇摆，既受惯性牵引又怕显得软弱时。",
                "unpacked_for_llm": "当你犹豫是继续投入还是及时退出时，先分开看沉没代价和不甘心。",
                "cross_domain_examples": [
                    {"domain": "daily_life", "scenario": "关系只剩消耗，却因投入太久不肯离开。"},
                    {"domain": "engineering", "scenario": "方案方向已错，却因前期投入巨大被强行延续。"},
                ],
            },
        )
        profile = build_activation_profile(wisdom)
        self.assertEqual(profile.family, "wisdom")
        self.assertFalse(profile.allow_lexical_fallback)

        relevant = ConditionedEvalRow(
            problem_id="p1",
            domain="daily_life",
            difficulty="medium",
            description="我已经投入很多钱和时间，不甘心退出，但继续下去身体越来越差。",
            coverage_tags=[],
            outcome="win",
            active_assumption_ids=["wisdom_W020"],
        )
        unrelated = ConditionedEvalRow(
            problem_id="p2",
            domain="business",
            difficulty="medium",
            description="如何给新产品制定渠道预算和首批客户画像。",
            coverage_tags=[],
            outcome="loss",
            active_assumption_ids=["wisdom_W020"],
        )

        self.assertEqual(route_problem_to_node(wisdom, relevant), RouteLabel.SHOULD_FIRE)
        self.assertEqual(route_problem_to_node(wisdom, unrelated), RouteLabel.NEUTRAL)

    def test_lifecycle_planner_maps_conditioned_gate_to_auditable_actions(self):
        summaries = [
            {
                "node_id": "strategy_S15",
                "claim": "incremental",
                "decision": "expand_retrieval",
                "route_counts": {"should_fire": 8},
                "active_counts": {"should_fire": 2},
                "active_should_fire_outcomes": {"win": 2},
                "utility_when_active_should_fire": 1.0,
                "utility_lcb90": 1.0,
                "harm_ucb90": None,
                "reasons": ["useful but under-retrieved"],
            },
            {
                "node_id": "strategy_S21",
                "claim": "stop dead end",
                "decision": "revise",
                "route_counts": {"should_fire": 6},
                "active_counts": {"should_fire": 6},
                "active_should_fire_outcomes": {"loss": 4, "win": 2},
                "utility_when_active_should_fire": 0.33,
                "utility_lcb90": 0.08,
                "harm_ucb90": None,
                "reasons": ["weak benefit"],
            },
        ]
        actions = plan_lifecycle_actions(summaries, eval_id="unit_eval")
        self.assertEqual(actions[0].action_type, LifecycleActionType.EXPAND_RETRIEVAL)
        self.assertEqual(actions[0].to_trial_manifest(eval_id="unit_eval").assumption_ids, ["strategy_S15"])
        self.assertEqual(actions[1].action_type, LifecycleActionType.REVISE_ASSUMPTION)

    def test_candidate_proposals_create_child_nodes_without_mutating_parent(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            store.upsert_node(AssumptionNode(
                id="strategy_S21",
                type=AssumptionType.METHOD,
                claim="识别当前路径已不可能成功，放弃并回溯到更高层决策点",
                tags=["S21"],
                confidence=0.8,
            ))
            store.flush()
            graph = SimpleAssumptionGraph(JsonlGraphStore(td))
            lifecycle_payload = {
                "actions": [
                    {
                        "node_id": "strategy_S21",
                        "action_type": "revise_assumption",
                        "priority": 0.7,
                        "rationale": "conditioned utility failed",
                        "proposed_updates": {"expected_effect": "child should beat parent"},
                        "verification_plan": "test child against parent",
                        "rollback_condition": "reject weak child",
                        "source": {"decision": "revise"},
                    }
                ]
            }

            proposals = build_candidate_proposals(
                graph=graph,
                lifecycle_payload=lifecycle_payload,
                eval_id="unit_eval",
            )
            self.assertEqual(len(proposals), 1)
            self.assertEqual(proposals[0].proposal_type, ProposalType.ASSUMPTION_REVISION)
            self.assertEqual(proposals[0].parent_node_id, "strategy_S21")
            self.assertEqual(proposals[0].candidate_node["status"], "candidate")
            self.assertIn("failure thresholds", proposals[0].candidate_node["claim"])
            self.assertIn("strategy_S21", graph.store.nodes)
            self.assertNotIn(proposals[0].candidate_node["id"], graph.store.nodes)

    def test_proposal_overlay_is_in_memory_only(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            store.upsert_node(AssumptionNode(
                id="strategy_S08",
                type=AssumptionType.METHOD,
                claim="提出猜测并测试",
                tags=["S08"],
            ))
            store.flush()
            graph = SimpleAssumptionGraph(JsonlGraphStore(td))
            lifecycle_payload = {
                "actions": [{
                    "node_id": "strategy_S08",
                    "action_type": "expand_retrieval",
                    "priority": 0.8,
                    "rationale": "useful but under-retrieved",
                    "proposed_updates": {"expected_effect": "increase trigger coverage"},
                    "verification_plan": "retrieval audit",
                    "rollback_condition": "outside harm",
                    "source": {
                        "decision": "expand_retrieval",
                        "utility_lcb90": 1.0,
                        "route_counts": {"should_fire": 4},
                        "active_counts": {"should_fire": 1},
                    },
                }]
            }
            proposals = build_candidate_proposals(graph=graph, lifecycle_payload=lifecycle_payload, eval_id="unit_eval")
            payload = {"proposals": [p.to_dict() for p in proposals]}

            overlay_store = JsonlGraphStore(td)
            applied = apply_proposal_overlay(overlay_store, payload)
            self.assertEqual(len(applied), 1)
            self.assertIn(applied[0], overlay_store.nodes)
            self.assertNotIn(applied[0], JsonlGraphStore(td).nodes)

    def test_candidate_eval_preflight_marks_ready_overlay(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = JsonlGraphStore(root / "graph")
            store.upsert_node(AssumptionNode(
                id="strategy_S08",
                type=AssumptionType.METHOD,
                claim="提出猜测并测试",
                tags=["S08", "假设检验"],
                confidence=0.7,
            ))
            store.flush()
            graph = SimpleAssumptionGraph(JsonlGraphStore(root / "graph"))
            lifecycle_payload = {
                "actions": [{
                    "node_id": "strategy_S08",
                    "action_type": "revise_assumption",
                    "priority": 0.7,
                    "rationale": "weak conditioned utility",
                    "proposed_updates": {"expected_effect": "child should beat parent"},
                    "verification_plan": "fresh ablation",
                    "rollback_condition": "reject weak child",
                    "source": {"decision": "revise", "utility_lcb90": 0.1},
                }]
            }
            proposals = build_candidate_proposals(graph=graph, lifecycle_payload=lifecycle_payload, eval_id="unit_eval")
            payload = {"eval_id": "unit_eval", "proposals": [p.to_dict() for p in proposals]}
            sample = [
                {
                    "problem_id": "p1",
                    "domain": "business",
                    "difficulty": "medium",
                    "description": "用一个低成本假设检验测试新渠道是否有效。",
                    "coverage_tags": ["S08"],
                },
                {
                    "problem_id": "p2",
                    "domain": "business",
                    "difficulty": "medium",
                    "description": "先提出可证伪假设，再用小样本测试。",
                    "coverage_tags": ["S08"],
                },
                {
                    "problem_id": "p3",
                    "domain": "engineering",
                    "difficulty": "medium",
                    "description": "对泵站故障提出可能原因并逐项测试。",
                    "coverage_tags": ["S08"],
                },
            ]
            meta = {p["problem_id"]: {"frame": "hybrid", "critical_reframe": "", "rewritten_problem": p["description"]} for p in sample}

            result = build_candidate_eval_payload(
                graph_dir=root / "graph",
                proposal_payload=payload,
                sample=sample,
                meta_by_pid=meta,
                eval_id="unit_preflight",
                min_trigger_n=3,
                min_active_trigger_n=2,
            )
            summary = result["summaries"][0]
            self.assertEqual(summary["readiness"], CandidateReadiness.READY_FOR_FRESH_ABLATION.value)
            self.assertGreaterEqual(len(summary["active_trigger_problem_ids"]), 2)

    def test_candidate_acceptance_gate_applies_only_accepted_children(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = JsonlGraphStore(root / "graph")
            store.upsert_node(AssumptionNode(
                id="strategy_S01",
                type=AssumptionType.METHOD,
                claim="control one variable",
                tags=["S01"],
            ))
            store.flush()
            graph = SimpleAssumptionGraph(JsonlGraphStore(root / "graph"))
            lifecycle_payload = {
                "actions": [{
                    "node_id": "strategy_S01",
                    "action_type": "revise_assumption",
                    "priority": 0.7,
                    "rationale": "weak conditioned utility",
                    "proposed_updates": {"expected_effect": "child should beat parent"},
                    "verification_plan": "fresh ablation",
                    "rollback_condition": "reject weak child",
                    "source": {"decision": "revise", "utility_lcb90": 0.1},
                }]
            }
            proposals = build_candidate_proposals(graph=graph, lifecycle_payload=lifecycle_payload, eval_id="unit_eval")
            proposal_payload = {"eval_id": "unit_eval", "proposals": [p.to_dict() for p in proposals]}
            proposal_id = proposals[0].proposal_id
            candidate_id = proposals[0].candidate_node["id"]
            preflight_payload = {
                "eval_id": "unit_preflight",
                "summaries": [{
                    "proposal_id": proposal_id,
                    "readiness": "ready_for_fresh_ablation",
                    "trigger_problem_ids": ["p1", "p2", "p3"],
                    "control_problem_ids": ["p4", "p5", "p6"],
                }],
            }
            judgment_path = root / "judgments.json"
            judgment_path.write_text(json.dumps({
                "p1": {"winner": "candidate"},
                "p2": {"winner": "candidate"},
                "p3": {"winner": "candidate"},
                "p4": {"winner": "tie"},
                "p5": {"winner": "candidate"},
                "p6": {"winner": "tie"},
            }), encoding="utf-8")

            acceptance = build_acceptance_payload(
                proposal_payload=proposal_payload,
                preflight_payload=preflight_payload,
                judgment_paths=[judgment_path],
                candidate_variant="candidate",
                baseline_variant="baseline",
                eval_id="unit_accept",
            )
            self.assertEqual(acceptance["accepted_proposal_ids"], [proposal_id])
            self.assertEqual(acceptance["summaries"][0]["decision"], AcceptanceDecision.ACCEPT.value)

            applied = apply_accepted_candidates(JsonlGraphStore(root / "graph"), proposal_payload, acceptance)
            self.assertEqual(applied, [candidate_id])
            updated = JsonlGraphStore(root / "graph")
            self.assertEqual(updated.nodes[candidate_id].status, "active")


if __name__ == "__main__":
    unittest.main()
