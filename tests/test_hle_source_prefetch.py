import json
import os
import tempfile
import urllib.error
import unittest
from pathlib import Path
from unittest.mock import patch

from assumption_os import hle_source_prefetch as prefetch
from assumption_os.diagnostic_logging import JsonlDiagnosticLogger


class TestHleSourcePrefetch(unittest.TestCase):
    def test_sanitize_problem_plan_drops_raw_queries(self):
        row = {
            "problem_id_hash": "problem-hash",
            "_stem": "SECRET RAW QUESTION",
            "_options": {"A": "SECRET OPTION"},
            "query_records": [
                {
                    "query_hash": "query-hash",
                    "query_kind": "option_claim",
                    "option_hash": "option-hash",
                    "option_label_hash": "option-hash",
                    "_query": "SECRET RAW QUERY",
                }
            ],
        }

        safe = prefetch._sanitize_problem_plan(row)
        serialized = json.dumps(safe)

        self.assertEqual(safe["query_count"], 1)
        self.assertEqual(safe["query_hashes"], ["query-hash"])
        self.assertEqual(safe["query_hashes_by_option_hash"], {"option-hash": ["query-hash"]})
        self.assertEqual(safe["query_kind_counts_by_option_hash"], {"option-hash": {"option_claim": 1}})
        self.assertNotIn("_query", serialized)
        self.assertNotIn("SECRET RAW QUERY", serialized)
        self.assertNotIn("SECRET RAW QUESTION", serialized)
        self.assertNotIn("SECRET OPTION", serialized)

    def test_sanitize_source_record_drops_internal_raw_fields(self):
        row = {
            "problem_id_hash": "problem-hash",
            "_query": "SECRET RAW QUERY",
            "_option_text": "SECRET OPTION",
            "_record_index": 2,
            "status": "fetched",
        }

        safe = prefetch._sanitize_source_record(row)
        serialized = json.dumps(safe)

        self.assertEqual(safe["status"], "fetched")
        self.assertNotIn("_query", serialized)
        self.assertNotIn("_option_text", serialized)
        self.assertNotIn("SECRET", serialized)

    def test_live_prefetch_env_temporarily_allows_source_search_then_restores(self):
        with patch.dict(os.environ, {"HLE_EVIDENCE_SOURCE_CACHE_ONLY": "1"}, clear=False):
            previous = prefetch._enter_prefetch_env(execute_live=True)
            try:
                self.assertEqual(os.environ.get("HLE_ALLOW_LIVE_SOURCE_SEARCH"), "1")
                self.assertNotIn("HLE_EVIDENCE_SOURCE_CACHE_ONLY", os.environ)
            finally:
                prefetch._restore_env(previous)

            self.assertEqual(os.environ.get("HLE_EVIDENCE_SOURCE_CACHE_ONLY"), "1")

    def test_run_source_prefetch_uses_live_fetch_only_on_cache_miss(self):
        query_plan = [
            {
                "problem_id_hash": "problem-hash",
                "seed_offset": 7,
                "operator_family_tags": ["controlled_variable"],
                "query_records": [
                    {
                        "query_hash": "query-hash",
                        "query_kind": "option_evidence",
                        "option_hash": "option-hash",
                        "option_label_hash": "option-hash",
                        "option_text_hash": "option-text-hash",
                        "option_choice": "A",
                        "_query": "raw query used only in memory",
                    }
                ],
            }
        ]

        with (
            patch.object(prefetch, "_cache_status", side_effect=["miss", "hit"]) as cache_status,
            patch.object(prefetch, "_fetch_source", return_value=[{"title": "t"}]) as fetch_source,
        ):
            records = prefetch._run_source_prefetch(
                query_plan=query_plan,
                sources=["semantic_scholar"],
                source_limit=2,
                timeout=1.0,
                execute_live=True,
                max_live_calls=1,
                delay_sec=0.0,
            )

        self.assertEqual(records[0]["status"], "fetched")
        self.assertEqual(records[0]["option_hash"], "option-hash")
        self.assertEqual(records[0]["option_label_hash"], "option-hash")
        self.assertEqual(records[0]["option_text_hash"], "option-text-hash")
        self.assertEqual(records[0]["option_choice"], "A")
        self.assertEqual(records[0]["row_count"], 1)
        self.assertEqual(records[0]["cache_status_after"], "hit")
        self.assertNotIn("_query", records[0])
        cache_status.assert_called()
        fetch_source.assert_called_once()

    def test_run_source_prefetch_logs_and_diagnoses_cache_hit_without_raw_text(self):
        option_hash = prefetch.stable_hash({"option_label": "A"})
        query_plan = [
            {
                "problem_id_hash": "problem-hash",
                "seed_offset": 7,
                "operator_family_tags": ["support_refute_evidence"],
                "_stem": "Which option directly supports the Alpha catalytic mechanism?",
                "_problem": {
                    "category": "Science",
                    "raw_subject": "Chemistry",
                },
                "query_records": [
                    {
                        "query_hash": "query-hash",
                        "query_kind": "option_claim",
                        "option_hash": option_hash,
                        "option_label_hash": option_hash,
                        "option_text_hash": prefetch.stable_hash({"option_text": "Alpha catalytic mechanism"}),
                        "option_choice": "A",
                        "_query": "Alpha catalytic mechanism support",
                        "_option_label": "A",
                        "_option_text": "Alpha catalytic mechanism",
                    }
                ],
            }
        ]
        rows = [
            {
                "title": "Alpha catalytic mechanism study",
                "snippet": "The Alpha catalytic mechanism directly supports the observed effect.",
                "source": "semantic_scholar",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "prefetch.jsonl"
            logger = JsonlDiagnosticLogger(log_path)
            with (
                patch.object(prefetch, "_cache_status", return_value="hit"),
                patch.object(prefetch, "_evidence_source_cache_get", return_value=rows),
            ):
                records = prefetch._run_source_prefetch(
                    query_plan=query_plan,
                    sources=["semantic_scholar"],
                    source_limit=2,
                    timeout=1.0,
                    execute_live=True,
                    max_live_calls=1,
                    delay_sec=0.0,
                    logger=logger,
                )

            self.assertEqual(records[0]["status"], "cache_hit")
            self.assertEqual(records[0]["option_hash"], option_hash)
            self.assertEqual(records[0]["answer_bearing_diagnostics_status"], "evaluated")
            self.assertGreater(records[0]["answer_bearing_option_signal_count"], 0)
            self.assertGreater(records[0]["answer_bearing_directish_count"], 0)
            serialized_record = json.dumps(records[0])
            self.assertNotIn("Alpha catalytic mechanism support", serialized_record)
            events = [json.loads(line) for line in log_path.read_text().splitlines()]

        event_names = {event["event"] for event in events}
        self.assertIn("hle_source_prefetch_record_planned", event_names)
        self.assertIn("hle_source_prefetch_cache_hit", event_names)
        self.assertIn("hle_source_prefetch_answer_bearing_diagnostics", event_names)
        self.assertTrue(all(event.get("option_hash") == option_hash for event in events))
        serialized_events = json.dumps(events)
        self.assertNotIn("Alpha catalytic mechanism support", serialized_events)
        self.assertNotIn("The Alpha catalytic mechanism directly supports", serialized_events)

    def test_prefetch_metrics_summarize_answer_bearing_by_option_hash(self):
        source_records = [
            {
                "option_hash": "option-a",
                "status": "cache_hit",
                "row_count": 2,
                "answer_bearing_diagnostics_status": "evaluated",
                "answer_bearing_directish_count": 1,
                "answer_bearing_option_signal_count": 2,
                "answer_bearing_best_score": 4.25,
            },
            {
                "option_hash": "option-b",
                "status": "cache_hit",
                "row_count": 1,
                "answer_bearing_diagnostics_status": "evaluated",
                "answer_bearing_directish_count": 0,
                "answer_bearing_option_signal_count": 1,
                "answer_bearing_best_score": 1.5,
            },
            {
                "option_hash": "option-a",
                "status": "dry_run_missing",
                "row_count": "bad",
                "answer_bearing_directish_count": "bad",
                "answer_bearing_option_signal_count": "",
                "answer_bearing_best_score": "bad",
            },
        ]

        metrics = prefetch._prefetch_metrics(query_plan=[], source_records=source_records)

        self.assertEqual(metrics["answer_bearing_directish_record_count"], 1)
        self.assertEqual(metrics["answer_bearing_directish_record_count_by_option_hash"], {"option-a": 1})
        self.assertEqual(
            metrics["answer_bearing_option_signal_record_count_by_option_hash"],
            {"option-a": 1, "option-b": 1},
        )
        self.assertEqual(
            metrics["answer_bearing_diagnostics_count_by_option_hash"],
            {"option-a": 1, "option-b": 1},
        )
        self.assertEqual(metrics["answer_bearing_source_row_count_by_option_hash"], {"option-a": 2, "option-b": 1})
        self.assertEqual(metrics["answer_bearing_best_score_max_by_option_hash"], {"option-a": 4.25, "option-b": 1.5})

    def test_run_source_prefetch_parallel_preserves_order_and_live_budget(self):
        query_plan = [
            {
                "problem_id_hash": "problem-hash",
                "seed_offset": 7,
                "operator_family_tags": [],
                "query_records": [
                    {
                        "query_hash": f"query-{index}",
                        "query_kind": "option_claim",
                        "_query": f"query {index}",
                    }
                    for index in range(3)
                ],
            }
        ]

        def cache_status(*, source, query, limit):
            if query == "query 0" and fetched_queries:
                return "hit"
            if query == "query 1" and query in fetched_queries:
                return "hit"
            return "miss"

        fetched_queries = set()

        def fetch_source(*, source, query, limit, timeout, ignore_cached_error=False):
            fetched_queries.add(query)
            return [{"title": query, "snippet": "row", "source": source}]

        with (
            patch.object(prefetch, "_cache_status", side_effect=cache_status),
            patch.object(prefetch, "_fetch_source", side_effect=fetch_source),
        ):
            records = prefetch._run_source_prefetch(
                query_plan=query_plan,
                sources=["semantic_scholar"],
                source_limit=2,
                timeout=1.0,
                execute_live=True,
                max_live_calls=2,
                delay_sec=0.0,
                parallel_workers=2,
            )

        self.assertEqual([row["query_hash"] for row in records], ["query-0", "query-1", "query-2"])
        self.assertEqual([row["status"] for row in records], ["fetched", "fetched", "budget_skipped"])
        self.assertEqual(len(fetched_queries), 2)

    def test_run_source_prefetch_round_robin_budget_covers_later_problems(self):
        query_plan = []
        for seed in [1, 2, 3]:
            query_plan.append(
                {
                    "problem_id_hash": f"problem-{seed}",
                    "seed_offset": seed,
                    "operator_family_tags": [],
                    "query_records": [
                        {
                            "query_hash": f"query-{seed}-{index}",
                            "query_kind": "option_claim",
                            "_query": f"query {seed} {index}",
                        }
                        for index in range(2)
                    ],
                }
            )

        def fetch_source(*, source, query, limit, timeout, ignore_cached_error=False):
            return [{"title": query, "snippet": "row", "source": source}]

        with (
            patch.object(prefetch, "_cache_status", return_value="miss"),
            patch.object(prefetch, "_fetch_source", side_effect=fetch_source),
        ):
            records = prefetch._run_source_prefetch(
                query_plan=query_plan,
                sources=["openalex"],
                source_limit=2,
                timeout=1.0,
                execute_live=True,
                max_live_calls=3,
                delay_sec=0.0,
                parallel_workers=1,
            )

        fetched_by_seed = [row["seed_offset"] for row in records if row["status"] == "fetched"]
        skipped_by_seed = [row["seed_offset"] for row in records if row["status"] == "budget_skipped"]
        self.assertEqual(fetched_by_seed, [1, 2, 3])
        self.assertEqual(skipped_by_seed, [1, 2, 3])

    def test_live_budget_covers_more_queries_before_duplicate_sources(self):
        query_plan = [
            {
                "problem_id_hash": "problem-1",
                "seed_offset": 1,
                "operator_family_tags": [],
                "query_records": [
                    {
                        "query_hash": f"query-{index}",
                        "query_kind": "option_claim",
                        "option_hash": "option-a",
                        "_query": f"query {index}",
                    }
                    for index in range(2)
                ],
            }
        ]

        def fetch_source(*, source, query, limit, timeout, ignore_cached_error=False):
            return [{"title": query, "snippet": "row", "source": source}]

        with (
            patch.object(prefetch, "_cache_status", return_value="miss"),
            patch.object(prefetch, "_fetch_source", side_effect=fetch_source),
        ):
            records = prefetch._run_source_prefetch(
                query_plan=query_plan,
                sources=["semantic_scholar", "openalex"],
                source_limit=2,
                timeout=1.0,
                execute_live=True,
                max_live_calls=2,
                delay_sec=0.0,
                parallel_workers=1,
            )

        self.assertEqual(
            [(row["query_hash"], row["source"], row["status"]) for row in records],
            [
                ("query-0", "semantic_scholar", "fetched"),
                ("query-0", "openalex", "budget_skipped"),
                ("query-1", "semantic_scholar", "fetched"),
                ("query-1", "openalex", "budget_skipped"),
            ],
        )

    def test_live_budget_round_robins_options_within_problem(self):
        query_plan = [
            {
                "problem_id_hash": "problem-1",
                "seed_offset": 1,
                "operator_family_tags": [],
                "query_records": [
                    {
                        "query_hash": "a-0",
                        "query_kind": "option_claim",
                        "option_hash": "option-a",
                        "_query": "A query 0",
                    },
                    {
                        "query_hash": "a-1",
                        "query_kind": "option_claim",
                        "option_hash": "option-a",
                        "_query": "A query 1",
                    },
                    {
                        "query_hash": "b-0",
                        "query_kind": "option_claim",
                        "option_hash": "option-b",
                        "_query": "B query 0",
                    },
                ],
            }
        ]
        fetch_order = []

        def fetch_source(*, source, query, limit, timeout, ignore_cached_error=False):
            fetch_order.append(query)
            return [{"title": query, "snippet": "row", "source": source}]

        with (
            patch.object(prefetch, "_cache_status", return_value="miss"),
            patch.object(prefetch, "_fetch_source", side_effect=fetch_source),
        ):
            records = prefetch._run_source_prefetch(
                query_plan=query_plan,
                sources=["openalex"],
                source_limit=2,
                timeout=1.0,
                execute_live=True,
                max_live_calls=2,
                delay_sec=0.0,
                parallel_workers=1,
            )

        self.assertEqual(fetch_order, ["A query 0", "B query 0"])
        self.assertEqual(
            [row["status"] for row in records],
            ["fetched", "budget_skipped", "fetched"],
        )

    def test_run_source_prefetch_executes_round_robin_fetches_before_front_loaded_static(self):
        query_plan = [
            {
                "problem_id_hash": "problem-1",
                "seed_offset": 1,
                "operator_family_tags": [],
                "query_records": [
                    {"query_hash": f"p1-{index}", "query_kind": "option_claim", "_query": f"p1 {index}"}
                    for index in range(4)
                ],
            },
            {
                "problem_id_hash": "problem-2",
                "seed_offset": 2,
                "operator_family_tags": [],
                "query_records": [
                    {"query_hash": "p2-0", "query_kind": "option_claim", "_query": "p2 0"}
                ],
            },
            {
                "problem_id_hash": "problem-3",
                "seed_offset": 3,
                "operator_family_tags": [],
                "query_records": [
                    {"query_hash": "p3-0", "query_kind": "option_claim", "_query": "p3 0"}
                ],
            },
        ]
        fetch_order = []

        def fetch_source(*, source, query, limit, timeout, ignore_cached_error=False):
            fetch_order.append(query)
            return [{"title": query, "snippet": "row", "source": source}]

        with (
            patch.object(prefetch, "_cache_status", return_value="miss"),
            patch.object(prefetch, "_fetch_source", side_effect=fetch_source),
        ):
            records = prefetch._run_source_prefetch(
                query_plan=query_plan,
                sources=["openalex"],
                source_limit=2,
                timeout=1.0,
                execute_live=True,
                max_live_calls=3,
                delay_sec=0.0,
                parallel_workers=1,
            )

        self.assertEqual(fetch_order, ["p1 0", "p2 0", "p3 0"])
        self.assertEqual(
            [row["status"] for row in records],
            ["fetched", "budget_skipped", "budget_skipped", "budget_skipped", "fetched", "fetched"],
        )

    def test_run_source_prefetch_source_error_budget_skips_later_source_fetches(self):
        query_plan = [
            {
                "problem_id_hash": "problem-1",
                "seed_offset": 1,
                "operator_family_tags": [],
                "query_records": [
                    {"query_hash": f"query-{index}", "query_kind": "option_claim", "_query": f"query {index}"}
                    for index in range(3)
                ],
            }
        ]

        def fetch_source(*, source, query, limit, timeout, ignore_cached_error=False):
            raise RuntimeError("source down")

        with (
            patch.object(prefetch, "_cache_status", return_value="miss"),
            patch.object(prefetch, "_fetch_source", side_effect=fetch_source),
        ):
            records = prefetch._run_source_prefetch(
                query_plan=query_plan,
                sources=["openalex"],
                source_limit=2,
                timeout=1.0,
                execute_live=True,
                max_live_calls=3,
                delay_sec=0.0,
                parallel_workers=1,
                source_error_budget=1,
            )

        self.assertEqual(
            [row["status"] for row in records],
            ["error", "source_error_budget_skipped", "source_error_budget_skipped"],
        )

    def test_run_source_prefetch_logs_sanitized_source_error_label(self):
        query_plan = [
            {
                "problem_id_hash": "problem-hash",
                "seed_offset": 7,
                "operator_family_tags": [],
                "query_records": [
                    {
                        "query_hash": "query-hash",
                        "query_kind": "option_claim",
                        "option_hash": "option-a",
                        "_query": "raw secret query",
                    }
                ],
            }
        ]

        def fetch_source(*, source, query, limit, timeout, ignore_cached_error=False):
            raise urllib.error.HTTPError(
                url="https://example.invalid/raw-secret-query",
                code=429,
                msg="Too Many Requests",
                hdrs=None,
                fp=None,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "prefetch.jsonl"
            logger = JsonlDiagnosticLogger(log_path)
            with (
                patch.object(prefetch, "_cache_status", return_value="miss"),
                patch.object(prefetch, "_fetch_source", side_effect=fetch_source),
            ):
                records = prefetch._run_source_prefetch(
                    query_plan=query_plan,
                    sources=["openalex"],
                    source_limit=2,
                    timeout=1.0,
                    execute_live=True,
                    max_live_calls=1,
                    delay_sec=0.0,
                    logger=logger,
                )
            events = [json.loads(line) for line in log_path.read_text().splitlines()]

        self.assertEqual(records[0]["status"], "error")
        self.assertEqual(records[0]["error_label"], "HTTPError_429")
        serialized = json.dumps({"records": records, "events": events})
        self.assertIn("HTTPError_429", serialized)
        self.assertNotIn("raw secret query", serialized)
        self.assertNotIn("raw-secret-query", serialized)

    def test_run_source_prefetch_skips_cached_error_without_live_fetch(self):
        query_plan = [
            {
                "problem_id_hash": "problem-hash",
                "seed_offset": 7,
                "operator_family_tags": ["controlled_variable"],
                "query_records": [
                    {
                        "query_hash": "query-hash",
                        "query_kind": "option_evidence",
                        "_query": "raw query used only in memory",
                    }
                ],
            }
        ]

        with (
            patch.object(prefetch, "_cache_status", return_value="cached_error"),
            patch.object(prefetch, "_fetch_source") as fetch_source,
        ):
            records = prefetch._run_source_prefetch(
                query_plan=query_plan,
                sources=["semantic_scholar"],
                source_limit=2,
                timeout=1.0,
                execute_live=True,
                max_live_calls=1,
                delay_sec=0.0,
            )

        self.assertEqual(records[0]["status"], "cached_error")
        self.assertFalse(records[0]["cached_error_retry_attempted"])
        fetch_source.assert_not_called()

    def test_run_source_prefetch_can_retry_cached_error_when_requested(self):
        query_plan = [
            {
                "problem_id_hash": "problem-hash",
                "seed_offset": 7,
                "operator_family_tags": ["controlled_variable"],
                "query_records": [
                    {
                        "query_hash": "query-hash",
                        "query_kind": "option_evidence",
                        "_query": "raw query used only in memory",
                    }
                ],
            }
        ]

        with (
            patch.object(prefetch, "_cache_status", side_effect=["cached_error", "hit"]),
            patch.object(prefetch, "_fetch_source", return_value=[{"title": "t"}]) as fetch_source,
        ):
            records = prefetch._run_source_prefetch(
                query_plan=query_plan,
                sources=["semantic_scholar"],
                source_limit=2,
                timeout=1.0,
                execute_live=True,
                max_live_calls=1,
                delay_sec=0.0,
                retry_cached_errors=True,
            )

        self.assertEqual(records[0]["status"], "fetched")
        self.assertEqual(records[0]["cache_status_before"], "cached_error")
        self.assertEqual(records[0]["cache_status_after"], "hit")
        self.assertTrue(records[0]["cached_error_retry_attempted"])
        self.assertEqual(
            records[0]["cached_error_retry_policy"],
            "ignore_cached_error_for_live_prefetch",
        )
        fetch_source.assert_called_once()
        self.assertTrue(fetch_source.call_args.kwargs["ignore_cached_error"])

    def test_fetch_source_temporarily_ignores_cached_error(self):
        def fake_search(query, *, limit, timeout):
            self.assertEqual(os.environ.get("HLE_SOURCE_PREFETCH_RETRY_CACHED_ERRORS"), "1")
            return [{"title": query, "source": "semantic_scholar"}]

        with patch.dict(os.environ, {}, clear=True):
            with patch.object(prefetch, "_semantic_scholar_search", side_effect=fake_search):
                rows = prefetch._fetch_source(
                    source="semantic_scholar",
                    query="direct relation",
                    limit=2,
                    timeout=1.0,
                    ignore_cached_error=True,
                )
            self.assertNotIn("HLE_SOURCE_PREFETCH_RETRY_CACHED_ERRORS", os.environ)

        self.assertEqual(rows[0]["title"], "direct relation")

    def test_problem_query_records_can_include_planner_queries_without_persisting_raw_text(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Genetics",
        }
        options = {"A": "Alpha", "B": "Beta mechanism"}
        planner_summary = {
            "status": "activated",
            "query_count": 1,
            "query_hashes_by_option_hash": {
                prefetch.stable_hash({"option_label": "B"}): [
                    prefetch.stable_hash({"query": "Beta mechanism direct relation"})
                ]
            },
            "underlying_model_calls": 1,
        }

        with patch.object(
            prefetch,
            "_run_option_claim_relation_query_planner",
            return_value=({"B": ["Beta mechanism direct relation"]}, planner_summary),
        ):
            records, summary = prefetch._problem_query_records(
                problem=problem,
                stem="Which mechanism directly explains the endpoint?",
                options=options,
                agent_plan={"stages": {}},
                max_options=2,
                max_queries_per_problem=4,
                max_queries_per_option=2,
                enable_relation_query_planner=True,
                relation_query_planner_model="gpt-5.4-mini",
        )

        self.assertEqual(summary["status"], "activated")
        self.assertIn("option_claim_relation_planner", [record["query_kind"] for record in records])
        safe = prefetch._sanitize_problem_plan({"problem_id_hash": "pid", "query_records": records})
        serialized = json.dumps(safe)
        self.assertIn("option_claim_relation_planner", serialized)
        self.assertNotIn("Beta mechanism direct relation", serialized)

    def test_problem_query_records_include_answer_web_fallback_queries(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Humanities/Social Science",
            "raw_subject": "Education",
        }
        options = {"A": "Alpha mitigation", "B": "Beta salient advice"}

        records, summary = prefetch._problem_query_records(
            problem=problem,
            stem="Which measure will NOT reduce automation bias?",
            options=options,
            agent_plan={"stages": {}},
            max_options=2,
            max_queries_per_problem=6,
            max_queries_per_option=3,
            enable_relation_query_planner=False,
            relation_query_planner_model="gpt-5.4-mini",
        )

        self.assertEqual(summary["status"], "disabled")
        self.assertIn("answer_web_fallback", [record["query_kind"] for record in records])
        safe = prefetch._sanitize_problem_plan({"problem_id_hash": "pid", "query_records": records})
        serialized = json.dumps(safe)
        self.assertIn("answer_web_fallback", serialized)
        self.assertNotIn("Beta salient advice", serialized)

    def test_option_aware_source_prefetch_queries_use_option_and_question_anchors(self):
        problem = {
            "id_hash": "pid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Immunology",
        }

        pairs = prefetch._option_aware_source_prefetch_queries(
            stem=(
                "Which antibody mechanism explains reduced binding to a glycosylated "
                "threonine repeat sequence?"
            ),
            option_text="Loss of epitope recognition caused by O-linked glycan shielding",
            problem=problem,
            agent_plan={"stages": {}},
        )

        kinds = [kind for kind, _query in pairs]
        queries = [query for _kind, query in pairs]
        self.assertIn("option_anchor_relation", kinds)
        self.assertTrue(any("epitope" in query.lower() for query in queries))
        self.assertTrue(any("Immunology" in query for query in queries))
        self.assertTrue(all("gold" not in query.lower() for query in queries))

    def test_problem_query_records_skip_option_aware_queries_by_default(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Molecular Biology",
        }
        options = {
            "A": "Alpha transporter alters sucrose uptake",
            "B": "Beta transporter changes raffinose secretion",
        }

        records, _summary = prefetch._problem_query_records(
            problem=problem,
            stem="Which mechanism explains altered aphid feeding under controlled sucrose conditions?",
            options=options,
            agent_plan={"stages": {}},
            max_options=2,
            max_queries_per_problem=8,
            max_queries_per_option=4,
            enable_relation_query_planner=False,
            relation_query_planner_model="gpt-5.4-mini",
        )

        kinds = [record["query_kind"] for record in records]
        self.assertNotIn("option_anchor_relation", kinds)
        self.assertNotIn("option_focus_phrase", kinds)
        self.assertNotIn("question_relation_anchor", kinds)

    def test_problem_query_records_include_option_aware_queries_when_enabled_without_raw_persistence(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Molecular Biology",
        }
        options = {
            "A": "Alpha transporter alters sucrose uptake",
            "B": "Beta transporter changes raffinose secretion",
        }

        records, _summary = prefetch._problem_query_records(
            problem=problem,
            stem="Which mechanism explains altered aphid feeding under controlled sucrose conditions?",
            options=options,
            agent_plan={"stages": {}},
            max_options=2,
            max_queries_per_problem=8,
            max_queries_per_option=4,
            enable_relation_query_planner=False,
            enable_option_aware_query_expansion=True,
            relation_query_planner_model="gpt-5.4-mini",
        )

        kinds = [record["query_kind"] for record in records]
        self.assertIn("option_anchor_relation", kinds)
        safe = prefetch._sanitize_problem_plan({"problem_id_hash": "pid", "query_records": records})
        serialized = json.dumps(safe)
        self.assertIn("option_anchor_relation", serialized)
        self.assertNotIn("Alpha transporter alters sucrose uptake", serialized)
        self.assertNotIn("controlled sucrose conditions", serialized)
        self.assertNotIn("_query", serialized)

    def test_problem_query_records_can_include_sweep_gap_relation_backfill_queries(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Computer Science/AI",
            "raw_subject": "Software Engineering",
        }
        options = {"A": "Alpha method", "B": "Beta target"}

        records, summary = prefetch._problem_query_records(
            problem=problem,
            stem="Which option preserves the migration dependency in a replacement benchmark?",
            options=options,
            agent_plan={"stages": {}},
            max_options=2,
            max_queries_per_problem=8,
            max_queries_per_option=4,
            enable_relation_query_planner=False,
            enable_sweep_gap_relation_backfill_queries=True,
            relation_query_planner_model="gpt-5.4-mini",
        )

        self.assertEqual(summary["status"], "disabled")
        kinds = [record["query_kind"] for record in records]
        self.assertIn("option_claim_deterministic_relation", kinds)
        self.assertTrue(
            {
                "option_claim_deterministic_relation",
                "option_claim_local_relation_expansion",
            }
            & set(kinds)
        )
        safe = prefetch._sanitize_problem_plan({"problem_id_hash": "pid", "query_records": records})
        serialized = json.dumps(safe)
        self.assertIn("option_claim_deterministic_relation", serialized)
        self.assertNotIn("Beta target", serialized)
        self.assertNotIn("_query", serialized)

    def test_balanced_prefetch_query_mix_includes_option_aware_when_no_relation_pressure(self):
        pairs = prefetch._balanced_prefetch_query_mix(
            relation_queries=[],
            sweep_gap_relation_queries=[],
            answer_web_queries=["answer web direct relation"],
            option_queries=["option evidence phrase"],
            claim_queries=["option claim answer phrase"],
            option_aware_query_pairs=[("option_anchor_relation", "option anchor relation phrase")],
            max_queries=4,
        )

        kinds = [kind for kind, _query in pairs]

        self.assertEqual(len(pairs), 4)
        self.assertEqual(
            kinds,
            ["answer_web_fallback", "option_claim", "option_evidence", "option_anchor_relation"],
        )

    def test_balanced_prefetch_query_mix_keeps_answer_bearing_families_under_backfill_pressure(self):
        pairs = prefetch._balanced_prefetch_query_mix(
            relation_queries=[],
            sweep_gap_relation_queries=[
                ("option_claim_deterministic_relation", f"deterministic relation {index}")
                for index in range(4)
            ]
            + [
                ("option_claim_local_relation_expansion", f"local relation {index}")
                for index in range(4)
            ],
            answer_web_queries=["answer web direct relation"],
            option_queries=["option evidence phrase"],
            claim_queries=["option claim answer phrase"],
            max_queries=5,
        )

        kinds = [kind for kind, _query in pairs]

        self.assertEqual(len(pairs), 5)
        self.assertIn("option_claim_deterministic_relation", kinds)
        self.assertIn("option_claim_local_relation_expansion", kinds)
        self.assertIn("answer_web_fallback", kinds)
        self.assertIn("option_claim", kinds)
        self.assertIn("option_evidence", kinds)

    def test_problem_query_records_balances_backfill_with_claim_and_answer_web_queries(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Medicine",
        }
        options = {"A": "Alpha diagnosis"}

        with (
            patch.object(
                prefetch,
                "_deterministic_option_claim_relation_queries",
                return_value=[f"deterministic relation {index}" for index in range(4)],
            ),
            patch.object(
                prefetch,
                "_option_claim_local_relation_query_expansion_queries",
                return_value=[f"local relation {index}" for index in range(4)],
            ),
            patch.object(
                prefetch,
                "_option_claim_answer_web_fallback_queries",
                return_value=["answer web direct relation"],
            ),
            patch.object(
                prefetch,
                "_option_claim_evidence_queries_for_plan",
                return_value=["option claim answer phrase"],
            ),
            patch.object(
                prefetch,
                "_option_evidence_queries_for_plan",
                return_value=["option evidence phrase"],
            ),
            patch.object(prefetch, "_option_claim_relation_slot_plan", return_value={}),
        ):
            records, _summary = prefetch._problem_query_records(
                problem=problem,
                stem="Which diagnosis explains the clinical endpoint?",
                options=options,
                agent_plan={"stages": {}},
                max_options=1,
                max_queries_per_problem=5,
                max_queries_per_option=5,
                enable_relation_query_planner=False,
                enable_sweep_gap_relation_backfill_queries=True,
                relation_query_planner_model="gpt-5.4-mini",
            )

        kinds = [record["query_kind"] for record in records]

        self.assertEqual(len(records), 5)
        self.assertIn("option_claim_deterministic_relation", kinds)
        self.assertIn("option_claim_local_relation_expansion", kinds)
        self.assertIn("answer_web_fallback", kinds)
        self.assertIn("option_claim", kinds)
        self.assertIn("option_evidence", kinds)

    def test_fetch_source_supports_answer_web(self):
        with patch.object(
            prefetch,
            "_answer_bearing_web_search",
            return_value=[{"title": "Beta", "source": "answer_web"}],
        ) as search:
            rows = prefetch._fetch_source(
                source="answer_web",
                query="Beta direct relation",
                limit=2,
                timeout=1.0,
            )

        self.assertEqual(rows[0]["source"], "answer_web")
        search.assert_called_once()


if __name__ == "__main__":
    unittest.main()
