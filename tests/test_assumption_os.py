import json
import tempfile
import unittest
from pathlib import Path

from assumption_os.adapters import ingest_artifacts, load_exp82_hypotheses, load_wisdom_nodes
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


if __name__ == "__main__":
    unittest.main()
