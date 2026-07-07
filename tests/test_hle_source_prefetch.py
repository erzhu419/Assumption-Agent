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
    def test_private_env_status_artifact_sanitizer_omits_key_names_and_path(self):
        safe = prefetch._sanitize_private_env_status_for_artifact({
            "loaded": True,
            "loaded_keys": ["OPENAI_API_KEY", "SEMANTIC_SCHOLAR_API_KEY"],
            "mode": "0o600",
            "path": "/home/user/.config/private.env",
        })

        serialized = json.dumps(safe)

        self.assertTrue(safe["loaded"])
        self.assertEqual(safe["loaded_key_count"], 2)
        self.assertIn("path_hash", safe)
        self.assertFalse(safe["raw_key_names_persisted"])
        self.assertFalse(safe["raw_path_persisted"])
        self.assertNotIn("OPENAI_API_KEY", serialized)
        self.assertNotIn("SEMANTIC_SCHOLAR_API_KEY", serialized)
        self.assertNotIn("/home/user/.config/private.env", serialized)

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

    def test_live_prefetch_env_can_temporarily_bypass_success_cache_reads(self):
        with patch.dict(os.environ, {}, clear=True):
            previous = prefetch._enter_prefetch_env(
                execute_live=True,
                refresh_cache_hits=True,
            )
            try:
                self.assertEqual(os.environ.get("HLE_ALLOW_LIVE_SOURCE_SEARCH"), "1")
                self.assertEqual(os.environ.get("HLE_EVIDENCE_SOURCE_CACHE_BYPASS_READ"), "1")
            finally:
                prefetch._restore_env(previous)

            self.assertNotIn("HLE_EVIDENCE_SOURCE_CACHE_BYPASS_READ", os.environ)

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

    def test_run_source_prefetch_respects_query_allowed_sources(self):
        query_plan = [
            {
                "problem_id_hash": "problem-hash",
                "seed_offset": 70,
                "operator_family_tags": ["answer_bearing_relation"],
                "query_records": [
                    {
                        "query_hash": "query-hash",
                        "query_kind": (
                            "candidate_specific_numeric_threshold_same_row_source_url_backfill"
                        ),
                        "option_hash": "option-hash",
                        "option_label_hash": "option-hash",
                        "_query": "https://example.org/xef4-preparation",
                        "allowed_sources": ["answer_web_fulltext"],
                    }
                ],
            }
        ]

        with (
            patch.object(prefetch, "_cache_status", return_value="miss") as cache_status,
            patch.object(prefetch, "_fetch_source", return_value=[{"title": "fulltext"}]) as fetch_source,
        ):
            records = prefetch._run_source_prefetch(
                query_plan=query_plan,
                sources=["semantic_scholar", "answer_web"],
                source_limit=1,
                timeout=1.0,
                execute_live=True,
                max_live_calls=3,
                delay_sec=0.0,
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source"], "answer_web_fulltext")
        self.assertEqual(cache_status.call_args.kwargs["source"], "answer_web_fulltext")
        self.assertEqual(fetch_source.call_args.kwargs["source"], "answer_web_fulltext")

    def test_run_source_prefetch_prioritizes_query_preferred_sources(self):
        query_plan = [
            {
                "problem_id_hash": "problem-hash",
                "seed_offset": 70,
                "operator_family_tags": ["answer_bearing_relation"],
                "query_records": [
                    {
                        "query_hash": "query-hash",
                        "query_kind": "candidate_specific_numeric_threshold_same_row_entity_anchor",
                        "option_hash": "option-hash",
                        "option_label_hash": "option-hash",
                        "_query": "BI-RADS assessment 4 minimum biopsy",
                        "preferred_sources": ["pubmed", "semantic_scholar"],
                    }
                ],
            }
        ]

        with (
            patch.object(prefetch, "_cache_status", return_value="miss"),
            patch.object(prefetch, "_fetch_source", return_value=[{"title": "pubmed"}]) as fetch_source,
        ):
            records = prefetch._run_source_prefetch(
                query_plan=query_plan,
                sources=["semantic_scholar", "pubmed"],
                source_limit=1,
                timeout=1.0,
                execute_live=True,
                max_live_calls=1,
                delay_sec=0.0,
            )

        fetched = [record for record in records if record["status"] == "fetched"]
        skipped = [record for record in records if record["status"] == "budget_skipped"]
        self.assertEqual(len(fetched), 1)
        self.assertEqual(fetched[0]["source"], "pubmed")
        self.assertEqual(fetch_source.call_args.kwargs["source"], "pubmed")
        self.assertEqual(len(skipped), 1)

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
                "numeric_same_row_diagnostics_status": "evaluated",
                "numeric_same_row_direct_count": 1,
                "numeric_same_row_value_match_count": 1,
                "numeric_same_row_best_score": 7.5,
                "numeric_same_row_rejection_reason_counts": {},
                "medical_guideline_permutation_ordering_status": "activated",
                "medical_guideline_permutation_ordering_candidate_exact": True,
                "medical_guideline_permutation_ordering_candidate_score": 17,
            },
            {
                "option_hash": "option-b",
                "status": "cache_hit",
                "row_count": 1,
                "answer_bearing_diagnostics_status": "evaluated",
                "answer_bearing_directish_count": 0,
                "answer_bearing_option_signal_count": 1,
                "answer_bearing_best_score": 1.5,
                "numeric_same_row_diagnostics_status": "evaluated",
                "numeric_same_row_direct_count": 0,
                "numeric_same_row_value_match_count": 1,
                "numeric_same_row_best_score": 4.1,
                "numeric_same_row_rejection_reason_counts": {
                    "relation_overlap_below_required": 1,
                },
                "fe_hyperfine_pair_binding_status": "evaluated",
                "fe_hyperfine_pair_binding_partial_row_count": 1,
                "fe_hyperfine_pair_binding_direct_row_count": 0,
                "fe_hyperfine_pair_binding_best_score": 6.0,
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
        self.assertEqual(metrics["numeric_same_row_diagnostics_evaluated_count"], 2)
        self.assertEqual(metrics["numeric_same_row_direct_record_count"], 1)
        self.assertEqual(metrics["numeric_same_row_value_match_record_count"], 2)
        self.assertEqual(
            metrics["numeric_same_row_direct_record_count_by_option_hash"],
            {"option-a": 1},
        )
        self.assertEqual(
            metrics["medical_guideline_permutation_ordering_status_counts"],
            {"activated": 1},
        )
        self.assertEqual(
            metrics["medical_guideline_permutation_ordering_exact_by_option_hash"],
            {"option-a": 1},
        )
        self.assertEqual(
            metrics["medical_guideline_permutation_ordering_unique_exact_option_hash"],
            "option-a",
        )
        self.assertEqual(
            metrics["medical_guideline_permutation_ordering_best_score_by_option_hash"],
            {"option-a": 17.0},
        )
        self.assertEqual(
            metrics["fe_hyperfine_pair_binding_status_counts"],
            {"evaluated": 1},
        )
        self.assertEqual(metrics["fe_hyperfine_pair_binding_partial_record_count"], 1)
        self.assertEqual(metrics["fe_hyperfine_pair_binding_direct_record_count"], 0)
        self.assertIsNone(metrics["fe_hyperfine_pair_binding_unique_direct_option_hash"])
        self.assertEqual(
            metrics["fe_hyperfine_pair_binding_best_score_by_option_hash"],
            {"option-b": 6.0},
        )
        self.assertEqual(
            metrics["numeric_same_row_value_match_record_count_by_option_hash"],
            {"option-a": 1, "option-b": 1},
        )
        self.assertEqual(metrics["numeric_same_row_best_score_max"], 7.5)
        self.assertEqual(
            metrics["numeric_same_row_rejection_reason_counts"],
            {"relation_overlap_below_required": 1},
        )

    def test_run_source_prefetch_records_numeric_same_row_diagnostics(self):
        option_hash = prefetch.stable_hash({"option_label": "B"})
        query_plan = [
            {
                "problem_id_hash": "problem-hash",
                "seed_offset": 70,
                "operator_family_tags": ["answer_bearing_relation"],
                "_stem": (
                    "Using any method of synthesis, which of the following is the coldest "
                    "temperature at which Xenon tetrafluoride can still be produced efficiently?"
                ),
                "_problem": {
                    "category": "Science",
                    "raw_subject": "Chemistry",
                },
                "query_records": [
                    {
                        "query_hash": "query-hash",
                        "query_kind": "candidate_specific_numeric_threshold_exact_value",
                        "option_hash": option_hash,
                        "option_label_hash": option_hash,
                        "option_text_hash": prefetch.stable_hash({"option_text": "400 C"}),
                        "option_choice": "B",
                        "_query": "xenon tetrafluoride 400 C lowest synthesis temperature",
                        "_option_label": "B",
                        "_option_text": "400 C",
                    }
                ],
            }
        ]
        rows = [
            {
                "title": "Xenon tetrafluoride preparation",
                "snippet": (
                    "XeF4, xenon tetrafluoride, has a lowest efficient synthesis "
                    "temperature near 400 C for the reaction of xenon and fluorine."
                ),
                "source": "semantic_scholar",
            }
        ]

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
            )

        self.assertEqual(records[0]["numeric_same_row_diagnostics_status"], "evaluated")
        self.assertEqual(records[0]["numeric_same_row_direct_count"], 1)
        self.assertEqual(records[0]["numeric_same_row_value_match_count"], 1)
        self.assertGreater(records[0]["numeric_same_row_best_score"], 0)
        serialized = json.dumps(records[0])
        self.assertNotIn("400 C", serialized)
        self.assertNotIn("Xenon tetrafluoride preparation", serialized)

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

    def test_live_budget_balances_sources_across_queries(self):
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
                ("query-1", "semantic_scholar", "budget_skipped"),
                ("query-1", "openalex", "fetched"),
            ],
        )

    def test_live_budget_rotates_first_source_across_problems(self):
        query_plan = []
        for seed in [1, 2, 3]:
            query_plan.append(
                {
                    "problem_id_hash": f"problem-{seed}",
                    "seed_offset": seed,
                    "operator_family_tags": [],
                    "query_records": [
                        {
                            "query_hash": f"query-{seed}",
                            "query_kind": "option_claim",
                            "_query": f"query {seed}",
                        }
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
                sources=["semantic_scholar", "openalex", "answer_web"],
                source_limit=2,
                timeout=1.0,
                execute_live=True,
                max_live_calls=3,
                delay_sec=0.0,
                parallel_workers=1,
            )

        self.assertEqual(
            [(row["seed_offset"], row["source"]) for row in records if row["status"] == "fetched"],
            [(1, "semantic_scholar"), (2, "openalex"), (3, "answer_web")],
        )

    def test_live_budget_rotates_first_option_across_problems(self):
        query_plan = []
        for seed in [1, 2, 3]:
            query_plan.append(
                {
                    "problem_id_hash": f"problem-{seed}",
                    "seed_offset": seed,
                    "operator_family_tags": [],
                    "query_records": [
                        {
                            "query_hash": f"query-{seed}-{label}",
                            "query_kind": "option_claim",
                            "option_hash": f"option-{label.lower()}",
                            "option_choice": label,
                            "_query": f"query {seed} {label}",
                        }
                        for label in ["A", "B", "C"]
                    ],
                }
            )
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
                sources=["answer_web"],
                source_limit=2,
                timeout=1.0,
                execute_live=True,
                max_live_calls=3,
                delay_sec=0.0,
                parallel_workers=1,
            )

        self.assertEqual(fetch_order, ["query 1 A", "query 2 B", "query 3 C"])
        self.assertEqual(
            [(row["seed_offset"], row["option_choice"]) for row in records if row["status"] == "fetched"],
            [(1, "A"), (2, "B"), (3, "C")],
        )

    def test_run_source_prefetch_logs_live_budget_by_source_without_raw_text(self):
        query_plan = [
            {
                "problem_id_hash": "problem-1",
                "seed_offset": 1,
                "operator_family_tags": [],
                "query_records": [
                    {
                        "query_hash": f"query-{index}",
                        "query_kind": "option_claim",
                        "_query": f"secret query {index}",
                    }
                    for index in range(3)
                ],
            }
        ]

        def fetch_source(*, source, query, limit, timeout, ignore_cached_error=False):
            return [{"title": query, "snippet": "row", "source": source}]

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "prefetch.jsonl"
            logger = JsonlDiagnosticLogger(log_path)
            with (
                patch.object(prefetch, "_cache_status", return_value="miss"),
                patch.object(prefetch, "_fetch_source", side_effect=fetch_source),
            ):
                prefetch._run_source_prefetch(
                    query_plan=query_plan,
                    sources=["semantic_scholar", "openalex", "answer_web"],
                    source_limit=2,
                    timeout=1.0,
                    execute_live=True,
                    max_live_calls=3,
                    delay_sec=0.0,
                    parallel_workers=1,
                    logger=logger,
                )
            events = [json.loads(line) for line in log_path.read_text().splitlines()]

        budget_events = [
            event for event in events if event.get("event") == "hle_source_prefetch_live_budget_applied"
        ]
        self.assertEqual(len(budget_events), 1)
        self.assertEqual(
            budget_events[0]["selected_count_by_source"],
            {"semantic_scholar": 1, "openalex": 1, "answer_web": 1},
        )
        self.assertNotIn("secret query", json.dumps(budget_events))

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

    def test_live_budget_prioritizes_term_identity_missing_required_queries(self):
        query_plan = [
            {
                "problem_id_hash": "problem-1",
                "seed_offset": 1,
                "operator_family_tags": [],
                "query_records": [
                    {
                        "query_hash": "regular",
                        "query_kind": "option_claim",
                        "option_hash": "option-a",
                        "_query": "regular query",
                    },
                    {
                        "query_hash": "term-gap",
                        "query_kind": "term_identity_missing_required_single",
                        "option_hash": "option-a",
                        "_query": "term gap query",
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
                max_live_calls=1,
                delay_sec=0.0,
                parallel_workers=1,
            )

        self.assertEqual(fetch_order, ["term gap query"])
        self.assertEqual(
            [row["status"] for row in records],
            ["budget_skipped", "fetched"],
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

    def test_answer_bearing_binding_prefetch_queries_bind_option_to_relation(self):
        problem = {
            "id_hash": "pid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Immunology",
        }

        pairs = prefetch._answer_bearing_binding_source_prefetch_queries(
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
        self.assertIn("answer_bearing_relation_binding", kinds)
        self.assertTrue(any("epitope" in query.lower() for query in queries))
        self.assertTrue(any("mechanism" in query.lower() for query in queries))
        self.assertTrue(any("Immunology" in query for query in queries))
        self.assertTrue(all("gold" not in query.lower() for query in queries))

    def test_required_relation_completion_prefetch_queries_bind_required_terms(self):
        problem = {
            "id_hash": "pid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Systems Biology",
        }

        pairs = prefetch._required_relation_completion_source_prefetch_queries(
            stem=(
                "Which mechanism preserves the controlled variable under "
                "replacement?"
            ),
            option_text="Beta mechanism",
            problem=problem,
            agent_plan={"stages": {}},
        )

        kinds = [kind for kind, _query in pairs]
        queries = [query for _kind, query in pairs]
        self.assertIn("answer_bearing_required_relation_completion", kinds)
        self.assertIn("answer_bearing_required_relation_exact_option", kinds)
        self.assertIn("answer_bearing_required_relation_term_pair", kinds)
        self.assertTrue(any("controlled" in query.lower() for query in queries))
        self.assertTrue(any("variable" in query.lower() for query in queries))
        self.assertTrue(any("Beta" in query for query in queries))
        self.assertTrue(all("gold" not in query.lower() for query in queries))

    def test_term_identity_missing_required_prefetch_queries_use_missing_term_hash(self):
        problem = {
            "id_hash": "pid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Systems Biology",
        }
        stem = "Which mechanism preserves the controlled variable under replacement?"
        option_text = "Beta mechanism"
        signature = prefetch._option_claim_question_relation_signature_terms(
            stem=stem,
            option_text=option_text,
            max_terms=12,
        )
        required_terms = [
            str(term).lower().strip("._-")
            for term in signature.get("required_terms", []) or []
            if str(term).strip("._-")
        ]
        missing_term = required_terms[0]
        missing_hash = prefetch.stable_hash({"relation_signature_term": missing_term})

        pairs = prefetch._term_identity_missing_required_source_prefetch_queries(
            stem=stem,
            option_text=option_text,
            problem=problem,
            agent_plan={"stages": {}},
            missing_required_term_hashes={missing_hash},
        )

        kinds = [kind for kind, _query in pairs]
        queries = [query for _kind, query in pairs]
        self.assertIn("term_identity_missing_required_single", kinds)
        self.assertTrue(any(missing_term in query.lower() for query in queries))
        self.assertTrue(any("Beta" in query for query in queries))
        self.assertTrue(all("gold" not in query.lower() for query in queries))

    def test_term_identity_missing_required_prefetch_queries_accept_all_terms_sentinel(self):
        problem = {
            "id_hash": "pid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Systems Biology",
        }
        stem = "Which mechanism preserves the controlled variable under replacement?"
        option_text = "Beta mechanism"
        signature = prefetch._option_claim_question_relation_signature_terms(
            stem=stem,
            option_text=option_text,
            max_terms=12,
        )
        required_terms = [
            str(term).lower().strip("._-")
            for term in signature.get("required_terms", []) or []
            if str(term).strip("._-")
        ]

        pairs = prefetch._term_identity_missing_required_source_prefetch_queries(
            stem=stem,
            option_text=option_text,
            problem=problem,
            agent_plan={"stages": {}},
            missing_required_term_hashes={
                prefetch._TERM_IDENTITY_ALL_REQUIRED_TERMS_SENTINEL,
            },
        )

        kinds = [kind for kind, _query in pairs]
        queries = [query for _kind, query in pairs]
        self.assertIn("term_identity_missing_required_single", kinds)
        self.assertTrue(
            any(term in query.lower() for term in required_terms for query in queries)
        )
        self.assertTrue(any("Beta" in query for query in queries))
        self.assertTrue(all("gold" not in query.lower() for query in queries))

    def test_term_identity_missing_required_prefetch_queries_use_stem_anchor_without_option_anchor(self):
        problem = {
            "id_hash": "pid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Systems Biology",
        }
        stem = "Which CRISPR-Cas9 mechanism preserves the controlled variable under replacement?"
        signature = prefetch._option_claim_question_relation_signature_terms(
            stem=stem,
            option_text="",
            max_terms=12,
        )
        required_terms = [
            str(term).lower().strip("._-")
            for term in signature.get("required_terms", []) or []
            if str(term).strip("._-")
        ]
        missing_hash = prefetch.stable_hash({
            "relation_signature_term": required_terms[0],
        })

        pairs = prefetch._term_identity_missing_required_source_prefetch_queries(
            stem=stem,
            option_text="",
            problem=problem,
            agent_plan={"stages": {}},
            missing_required_term_hashes={missing_hash},
        )

        kinds = [kind for kind, _query in pairs]
        queries = [query for _kind, query in pairs]
        self.assertIn("term_identity_missing_required_stem_anchor", kinds)
        self.assertTrue(any("CRISPR" in query for query in queries))
        self.assertTrue(any(required_terms[0] in query.lower() for query in queries))
        self.assertTrue(all("gold" not in query.lower() for query in queries))

    def test_query_has_option_anchor_accepts_numeric_option_terms(self):
        self.assertTrue(
            prefetch._source_prefetch_query_has_option_anchor(
                query="78 xenon tetrafluoride temperature Chemistry",
                option_text="-78 C",
            )
        )
        self.assertFalse(
            prefetch._source_prefetch_query_has_option_anchor(
                query="400 xenon tetrafluoride temperature Chemistry",
                option_text="-78 C",
            )
        )
        self.assertTrue(
            prefetch._source_prefetch_query_has_option_anchor(
                query="xenon tetrafluoride synthesis temperature 195 K",
                option_text="-78 C",
            )
        )
        self.assertFalse(
            prefetch._source_prefetch_query_has_option_anchor(
                query="xenon tetrafluoride synthesis temperature 400 K",
                option_text="-78 C",
            )
        )

    def test_source_failure_focus_extracts_missing_required_term_hashes(self):
        option_hash = prefetch.stable_hash({"option_label": "B"})
        payload = {
            "eval_id": "eval",
            "sampling": {"seed_offset": 1079},
            "rows": [
                {
                    "component_efficacy": {
                        "mc_option_claim_evidence_verifier": {
                            "candidate_direct_relation_span_required_coverage_gap_rows": [
                                {
                                    "option_hash": option_hash,
                                    "top_required_missing_count": 2,
                                    "top_required_missing_term_hashes": [
                                        "term-hash-1",
                                        "term-hash-2",
                                    ],
                                }
                            ]
                        }
                    }
                }
            ],
        }

        focus = prefetch._source_failure_focus_from_eval_payload(payload)

        seed_focus = focus["focus_by_seed"][1079]
        self.assertIn(option_hash, seed_focus["option_hashes"])
        self.assertEqual(
            seed_focus["option_missing_required_term_hashes"][option_hash],
            ["term-hash-1", "term-hash-2"],
        )
        self.assertIn("required_term_hash_gap", focus["reason_counts"])

    def test_source_failure_focus_extracts_span_directness_missing_required_all_terms(self):
        option_hash = prefetch.stable_hash({"option_label": "B"})
        payload = {
            "eval_id": "eval",
            "sampling": {"seed_offset": 1079},
            "rows": [
                {
                    "component_efficacy": {
                        "mc_option_claim_evidence_verifier": {
                            "span_directness_verifier_candidate_directness_rows": [
                                {
                                    "option_hash": option_hash,
                                    "programmatic_gap_reason": (
                                        "missing_required_relation_terms"
                                    ),
                                    "direct_high_confidence": False,
                                }
                            ]
                        }
                    }
                }
            ],
        }

        focus = prefetch._source_failure_focus_from_eval_payload(payload)

        seed_focus = focus["focus_by_seed"][1079]
        self.assertIn(option_hash, seed_focus["option_hashes"])
        self.assertEqual(
            seed_focus["option_missing_required_term_hashes"][option_hash],
            [prefetch._TERM_IDENTITY_ALL_REQUIRED_TERMS_SENTINEL],
        )
        self.assertIn("required_term_hash_gap", focus["reason_counts"])
        self.assertIn("required_term_identity_all_terms_gap", focus["reason_counts"])

    def test_problem_query_records_include_term_identity_missing_required_queries(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Systems Biology",
        }
        stem = "Which mechanism preserves the controlled variable under replacement?"
        options = {"A": "Alpha distractor", "B": "Beta mechanism"}
        option_hash = prefetch.stable_hash({"option_label": "B"})
        signature = prefetch._option_claim_question_relation_signature_terms(
            stem=stem,
            option_text=options["B"],
            max_terms=12,
        )
        missing_term = str(signature["required_terms"][0]).lower().strip("._-")
        missing_hash = prefetch.stable_hash({"relation_signature_term": missing_term})

        records, _summary = prefetch._problem_query_records(
            problem=problem,
            stem=stem,
            options=options,
            agent_plan={"stages": {}},
            max_options=2,
            max_queries_per_problem=6,
            max_queries_per_option=3,
            enable_relation_query_planner=False,
            relation_query_planner_model="gpt-5.4-mini",
            focus_missing_required_term_hashes_by_option={
                option_hash: {missing_hash},
            },
        )

        kinds = [record["query_kind"] for record in records]
        self.assertIn("term_identity_missing_required_single", kinds)
        safe = prefetch._sanitize_problem_plan(
            {"problem_id_hash": "pid", "query_records": records}
        )
        serialized = json.dumps(safe)
        self.assertIn("term_identity_missing_required_single", serialized)
        self.assertNotIn("Beta mechanism", serialized)

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
        self.assertNotIn("answer_bearing_relation_binding", kinds)
        self.assertNotIn("answer_bearing_option_focus", kinds)
        self.assertNotIn("answer_bearing_pair_binding", kinds)
        self.assertNotIn("answer_bearing_pair_contrast", kinds)
        self.assertNotIn("candidate_specific_answer_bearing_witness", kinds)

    def test_failure_focus_extracts_rejected_generic_candidate_hashes_without_raw_text(self):
        option_hash = prefetch.stable_hash({"option_label": "B"})
        payload = {
            "eval_id": "focus-eval",
            "sampling": {"seed_offset": 971},
            "rows": [
                {
                    "component_efficacy": {
                        "mc_option_claim_evidence_verifier": {
                            "span_directness_verifier_candidate_directness_rows": [
                                {
                                    "option_hash": option_hash,
                                    "evidence_relation": "generic",
                                    "lexical_unique_but_relation_generic": True,
                                    "candidate_relation_span_directness": {
                                        "programmatic_gap_gate": {
                                            "programmatic_gap_reason": "missing_required_relation_terms"
                                        }
                                    },
                                }
                            ],
                            "contrastive_adjudicator_structured_relation_matrix": [
                                {
                                    "option_hash": option_hash,
                                    "evidence_relation": "indirect",
                                    "source_verifier_rejection_reason": "no_selected_label_generic",
                                    "has_source_quality": True,
                                    "source_verified_direct": False,
                                }
                            ],
                        }
                    },
                    "_question": "SECRET RAW QUESTION",
                }
            ],
        }

        focus = prefetch._source_failure_focus_from_eval_payload(payload)
        safe = prefetch._sanitize_failure_focus_summary(focus)
        serialized = json.dumps(safe)

        self.assertEqual(focus["status"], "activated")
        self.assertEqual(focus["focused_option_count"], 1)
        self.assertEqual(safe["focused_option_hashes_by_seed"], {"971": [option_hash]})
        self.assertGreater(safe["reason_counts"]["span_directness_generic"], 0)
        self.assertGreater(safe["reason_counts"]["source_verifier_rejected_relation"], 0)
        self.assertNotIn("SECRET RAW QUESTION", serialized)
        self.assertNotIn("_question", serialized)

    def test_failure_focus_extracts_candidate_span_bundle_gap_options(self):
        direct_hash = prefetch.stable_hash({"option_label": "B"})
        indirect_hash = prefetch.stable_hash({"option_label": "C"})
        payload = {
            "eval_id": "focus-eval",
            "sampling": {"seed_offset": 971},
            "rows": [
                {
                    "component_efficacy": {
                        "mc_option_claim_evidence_verifier": {
                            "source_quality_directness_promotion_detail": {
                                "option_matrix_candidate_span_bundle_source_lane": {
                                    "status": "blocked",
                                    "reason": "candidate_span_bundle_no_strong_direct_margin",
                                    "option_summaries": [
                                        {
                                            "option_hash": direct_hash,
                                            "bundle_type": "direct_relation",
                                            "relation_established": True,
                                        },
                                        {
                                            "option_hash": indirect_hash,
                                            "bundle_type": "indirect",
                                            "relation_established": False,
                                        },
                                    ],
                                },
                                "option_matrix_source_audit_lane": {
                                    "status": "ambiguous",
                                    "reason": "direct_pair_bound_margin_too_small",
                                    "option_summaries": [
                                        {
                                            "option_hash": direct_hash,
                                            "bundle_type": "direct_relation",
                                            "relation_established": True,
                                        }
                                    ],
                                },
                            }
                        }
                    },
                    "_question": "SECRET RAW QUESTION",
                }
            ],
        }

        focus = prefetch._source_failure_focus_from_eval_payload(payload)
        safe = prefetch._sanitize_failure_focus_summary(focus)
        serialized = json.dumps(safe)

        self.assertEqual(focus["status"], "activated")
        self.assertEqual(
            safe["focused_option_hashes_by_seed"],
            {"971": [direct_hash, indirect_hash]},
        )
        self.assertEqual(
            safe["reason_counts"]["candidate_span_bundle_direct_ambiguous_margin"],
            1,
        )
        self.assertEqual(
            safe["reason_counts"][
                "candidate_span_bundle_indirect_needs_candidate_specific_source"
            ],
            1,
        )
        self.assertEqual(
            safe["reason_counts"]["source_audit_direct_ambiguous_margin"],
            1,
        )
        self.assertNotIn("SECRET RAW QUESTION", serialized)

    def test_failure_focus_marks_expand_all_options_when_gold_was_missed(self):
        option_hash = prefetch.stable_hash({"option_label": "B"})
        payload = {
            "eval_id": "focus-eval",
            "sampling": {"seed_offset": 971},
            "rows": [
                {
                    "component_efficacy": {
                        "candidate_generation_missed_gold": True,
                        "mc_option_claim_evidence_verifier": {
                            "span_directness_verifier_candidate_directness_rows": [
                                {
                                    "option_hash": option_hash,
                                    "evidence_relation": "generic",
                                }
                            ],
                        },
                    },
                    "_question": "SECRET RAW QUESTION",
                }
            ],
        }

        focus = prefetch._source_failure_focus_from_eval_payload(payload)
        safe = prefetch._sanitize_failure_focus_summary(focus)

        self.assertEqual(focus["status"], "activated")
        self.assertEqual(focus["expand_all_option_seed_count"], 1)
        self.assertEqual(safe["expand_all_options_by_seed"], {"971": True})
        self.assertIn("candidate_generation_missed_gold_expand_all_options", safe["reason_counts"])
        self.assertNotIn("SECRET RAW QUESTION", json.dumps(safe))

    def test_failure_focus_trim_keeps_highest_candidate_specific_signal(self):
        weak_hash = prefetch.stable_hash({"option_label": "A"})
        strong_hash = prefetch.stable_hash({"option_label": "B"})
        payload = {
            "eval_id": "focus-eval",
            "sampling": {"seed_offset": 971},
            "rows": [
                {
                    "component_efficacy": {
                        "mc_option_claim_evidence_verifier": {
                            "span_directness_verifier_candidate_directness_rows": [
                                {
                                    "option_hash": weak_hash,
                                    "evidence_relation": "generic",
                                },
                                {
                                    "option_hash": strong_hash,
                                    "evidence_relation": "generic",
                                    "candidate_relation_span_source_cache_targeted_near_complete_count": 1,
                                    "candidate_relation_span_directness": {
                                        "programmatic_gap_gate": {
                                            "programmatic_gap_reason": "missing_required_relation_terms"
                                        }
                                    },
                                },
                            ],
                            "contrastive_adjudicator_structured_relation_matrix": [
                                {
                                    "option_hash": weak_hash,
                                    "has_source_quality": True,
                                    "source_verified_direct": False,
                                }
                            ],
                        }
                    }
                }
            ],
        }

        focus = prefetch._source_failure_focus_from_eval_payload(payload)
        trimmed = prefetch._trim_source_failure_focus(focus, max_options_per_problem=1)
        safe = prefetch._sanitize_failure_focus_summary(trimmed)

        self.assertEqual(safe["focused_option_hashes_by_seed"], {"971": [strong_hash]})
        self.assertEqual(safe["focused_option_count"], 1)
        self.assertEqual(safe["untrimmed_focused_option_count"], 2)
        self.assertEqual(safe["focus_trim_policy"], "top_candidate_specific_failure_score_v1")

    def test_problem_query_records_focus_only_targets_rejected_generic_candidates(self):
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
            "C": "Gamma channel changes fructose loading",
        }
        focus_hash = prefetch.stable_hash({"option_label": "B"})

        records, _summary = prefetch._problem_query_records(
            problem=problem,
            stem="Which mechanism explains altered aphid feeding under controlled sucrose conditions?",
            options=options,
            agent_plan={"stages": {}},
            max_options=3,
            max_queries_per_problem=8,
            max_queries_per_option=4,
            enable_relation_query_planner=False,
            enable_candidate_specific_answer_bearing_queries=True,
            focus_option_hashes={focus_hash},
            focus_only=True,
            relation_query_planner_model="gpt-5.4-mini",
        )

        self.assertTrue(records)
        self.assertEqual({record["option_hash"] for record in records}, {focus_hash})
        self.assertIn(
            "candidate_specific_answer_bearing_witness",
            [record["query_kind"] for record in records],
        )
        safe = prefetch._sanitize_problem_plan({"problem_id_hash": "pid", "query_records": records})
        serialized = json.dumps(safe)
        self.assertIn("candidate_specific_answer_bearing_witness", serialized)
        self.assertNotIn("Beta transporter changes raffinose secretion", serialized)
        self.assertNotIn("controlled sucrose conditions", serialized)
        self.assertNotIn("_query", serialized)

    def test_problem_query_records_focus_only_expands_all_options_for_missed_gold(self):
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
            "C": "Gamma channel changes fructose loading",
        }
        focus_hash = prefetch.stable_hash({"option_label": "B"})

        records, _summary = prefetch._problem_query_records(
            problem=problem,
            stem="Which mechanism explains altered aphid feeding under controlled sucrose conditions?",
            options=options,
            agent_plan={"stages": {}},
            max_options=3,
            max_queries_per_problem=12,
            max_queries_per_option=4,
            enable_relation_query_planner=False,
            enable_candidate_specific_answer_bearing_queries=True,
            focus_option_hashes={focus_hash},
            focus_all_options=True,
            focus_only=True,
            relation_query_planner_model="gpt-5.4-mini",
        )

        self.assertEqual(
            {record["option_hash"] for record in records},
            {prefetch.stable_hash({"option_label": label}) for label in options},
        )
        safe = prefetch._sanitize_problem_plan({"problem_id_hash": "pid", "query_records": records})
        serialized = json.dumps(safe)
        self.assertNotIn("Gamma channel changes fructose loading", serialized)
        self.assertNotIn("_query", serialized)

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

    def test_problem_query_records_include_answer_bearing_binding_queries_when_enabled(self):
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
            enable_answer_bearing_binding_queries=True,
            relation_query_planner_model="gpt-5.4-mini",
        )

        kinds = [record["query_kind"] for record in records]
        self.assertIn("answer_bearing_relation_binding", kinds)
        safe = prefetch._sanitize_problem_plan({"problem_id_hash": "pid", "query_records": records})
        serialized = json.dumps(safe)
        self.assertIn("answer_bearing_relation_binding", serialized)
        self.assertNotIn("Alpha transporter alters sucrose uptake", serialized)
        self.assertNotIn("controlled sucrose conditions", serialized)
        self.assertNotIn("_query", serialized)

    def test_problem_query_records_include_required_relation_completion_queries_when_enabled(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Systems Biology",
        }
        options = {
            "A": "Alpha mechanism",
            "B": "Beta mechanism",
        }

        records, _summary = prefetch._problem_query_records(
            problem=problem,
            stem=(
                "Which mechanism preserves the controlled variable under "
                "replacement?"
            ),
            options=options,
            agent_plan={"stages": {}},
            max_options=2,
            max_queries_per_problem=8,
            max_queries_per_option=4,
            enable_relation_query_planner=False,
            enable_required_relation_completion_queries=True,
            relation_query_planner_model="gpt-5.4-mini",
        )

        kinds = [record["query_kind"] for record in records]
        self.assertIn("answer_bearing_required_relation_completion", kinds)
        safe = prefetch._sanitize_problem_plan({
            "problem_id_hash": "pid",
            "query_records": records,
        })
        serialized = json.dumps(safe)
        self.assertIn("answer_bearing_required_relation_completion", serialized)
        self.assertNotIn("Beta mechanism", serialized)
        self.assertNotIn("controlled variable", serialized)
        self.assertNotIn("_query", serialized)

    def test_answer_bearing_pair_binding_prefetch_queries_bind_two_options(self):
        problem = {
            "id_hash": "pid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Molecular Biology",
        }
        pairs = prefetch._answer_bearing_pair_binding_source_prefetch_queries(
            stem="Which mechanism explains altered aphid feeding under controlled sucrose conditions?",
            option_label="A",
            option_text="Alpha transporter alters sucrose uptake",
            options={
                "A": "Alpha transporter alters sucrose uptake",
                "B": "Beta transporter changes raffinose secretion",
            },
            problem=problem,
            agent_plan={"stages": {}},
        )

        self.assertTrue(pairs)
        kinds = [kind for kind, _query in pairs]
        queries = [query for _kind, query in pairs]
        self.assertIn("answer_bearing_pair_binding", kinds)
        self.assertTrue(any("Alpha" in query and "Beta" in query for query in queries))
        self.assertTrue(all("gold" not in query.lower() for query in queries))

    def test_candidate_specific_answer_bearing_prefetch_queries_bind_option_and_required_terms(self):
        problem = {
            "id_hash": "pid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Molecular Biology",
        }
        pairs = prefetch._candidate_specific_answer_bearing_source_prefetch_queries(
            stem=(
                "Which mechanism explains altered aphid feeding under controlled "
                "sucrose conditions?"
            ),
            option_label="A",
            option_text="Alpha transporter alters sucrose uptake",
            options={
                "A": "Alpha transporter alters sucrose uptake",
                "B": "Beta transporter changes raffinose secretion",
            },
            problem=problem,
            agent_plan={"stages": {}},
        )

        self.assertTrue(pairs)
        kinds = [kind for kind, _query in pairs]
        queries = [query for _kind, query in pairs]
        self.assertIn("candidate_specific_answer_bearing_witness", kinds)
        self.assertIn("candidate_specific_answer_bearing_exact_option", kinds)
        self.assertIn("candidate_specific_answer_bearing_required_pair", kinds)
        self.assertTrue(any("Alpha" in query for query in queries))
        self.assertTrue(
            any("controlled" in query.lower() or "sucrose" in query.lower() for query in queries)
        )
        self.assertTrue(any("Molecular" in query or "Biology" in query for query in queries))
        self.assertTrue(all("gold" not in query.lower() for query in queries))

    def test_patient_label_resolution_query_expands_option_labels_to_descriptions(self):
        pairs = prefetch._source_prefetch_patient_label_resolution_queries(
            stem=(
                "Patient 1: Severe burst fracture of L2, no neurologic deficits. "
                "Patient 2: Compression fracture of L2 with mild traumatic "
                "spondylolisthesis of L1, no neurologic deficits. "
                "Patient 3: Split fracture of L2, with mildly disordered pelvic "
                "functions. Prioritize according to surgical indications."
            ),
            option_text="Patient 3, Patient 2, Patient 1",
            problem={
                "answer_type": "multipleChoice",
                "category": "Biology/Medicine",
                "raw_subject": "Medicine",
            },
            relation_focus=["surgical", "indications", "priority"],
            subject_terms=["Medicine"],
        )

        self.assertTrue(pairs)
        self.assertTrue(
            all(
                kind == "candidate_specific_answer_bearing_patient_label_resolution"
                for kind, _query in pairs
            )
        )
        joined_queries = "\n".join(query for _kind, query in pairs)
        self.assertIn("Patient 3", joined_queries)
        self.assertIn("split fracture", joined_queries.lower())
        self.assertIn("pelvic", joined_queries.lower())
        self.assertIn("Patient 2", joined_queries)
        self.assertIn("spondylolisthesis", joined_queries.lower())
        self.assertIn("surgical", joined_queries.lower())
        self.assertNotIn("gold", joined_queries.lower())

    def test_oxidation_spin_alias_query_adds_fe_aliases_and_relation_terms(self):
        pairs = prefetch._source_prefetch_oxidation_spin_alias_queries(
            stem=(
                "Which combination has the largest hyperfine field in 57Fe "
                "Mossbauer spectroscopy?"
            ),
            option_text="planar S = 5/2 Fe(III)",
            problem={
                "answer_type": "multipleChoice",
                "category": "Chemistry",
                "raw_subject": "Chemistry",
            },
            relation_focus=["largest", "hyperfine", "field"],
            subject_terms=["Chemistry"],
        )

        self.assertTrue(pairs)
        kind, query = pairs[0]
        self.assertEqual(kind, "candidate_specific_answer_bearing_oxidation_spin_alias")
        self.assertTrue("Fe 3+" in query or "ferric" in query)
        self.assertTrue("S 5 2" in query or "spin 5 2" in query)
        self.assertIn("hyperfine", query.lower())
        self.assertIn("Mossbauer", query)
        self.assertNotIn("gold", query.lower())

    def test_problem_query_records_include_patient_and_oxidation_expansion_queries(self):
        patient_records, _summary = prefetch._problem_query_records(
            problem={
                "id_hash": "patient",
                "answer_type": "multipleChoice",
                "category": "Biology/Medicine",
                "raw_subject": "Medicine",
            },
            stem=(
                "Patient 1: Severe burst fracture of L2, no neurologic deficits. "
                "Patient 2: Compression fracture of L2 with mild traumatic "
                "spondylolisthesis of L1, no neurologic deficits. "
                "Patient 3: Split fracture of L2, with mildly disordered pelvic "
                "functions. Prioritize according to surgical indications."
            ),
            options={
                "A": "Patient 3, Patient 2, Patient 1",
                "B": "Patient 1, Patient 2, Patient 3",
            },
            agent_plan={"stages": {}},
            max_options=2,
            max_queries_per_problem=8,
            max_queries_per_option=4,
            enable_candidate_specific_answer_bearing_queries=True,
        )
        patient_kinds = {record.get("query_kind") for record in patient_records}
        self.assertIn(
            "candidate_specific_answer_bearing_patient_label_resolution",
            patient_kinds,
        )
        patient_expansion_records = [
            record
            for record in patient_records
            if record.get("query_kind")
            == "candidate_specific_answer_bearing_patient_label_resolution"
        ]
        self.assertTrue(patient_expansion_records)
        self.assertTrue(
            any(
                "split fracture" in str(record.get("_source_diagnostic_option_text") or "").lower()
                for record in patient_expansion_records
            )
        )
        self.assertTrue(
            all(
                record.get("source_diagnostic_option_text_hash")
                for record in patient_expansion_records
            )
        )

        oxidation_records, _summary = prefetch._problem_query_records(
            problem={
                "id_hash": "iron",
                "answer_type": "multipleChoice",
                "category": "Chemistry",
                "raw_subject": "Chemistry",
            },
            stem=(
                "Which combination has the largest hyperfine field in 57Fe "
                "Mossbauer spectroscopy?"
            ),
            options={
                "A": "square pyramidal S = 0 Fe(II)",
                "B": "planar S = 5/2 Fe(III)",
            },
            agent_plan={"stages": {}},
            max_options=2,
            max_queries_per_problem=8,
            max_queries_per_option=4,
            enable_candidate_specific_answer_bearing_queries=True,
        )
        oxidation_kinds = {record.get("query_kind") for record in oxidation_records}
        self.assertIn(
            "candidate_specific_answer_bearing_oxidation_spin_alias",
            oxidation_kinds,
        )
        oxidation_expansion_records = [
            record
            for record in oxidation_records
            if record.get("query_kind")
            == "candidate_specific_answer_bearing_oxidation_spin_alias"
        ]
        self.assertTrue(
            any(
                "ferric" in str(record.get("_source_diagnostic_option_text") or "").lower()
                for record in oxidation_expansion_records
            )
        )

    def test_source_diagnostics_use_expanded_patient_label_claim(self):
        detail = prefetch._source_rows_answer_bearing_diagnostics(
            rows=[
                {
                    "title": "Thoracolumbar injury classification",
                    "snippet": (
                        "A severe burst fracture with posterior ligamentous "
                        "complex injury is associated with surgical indications "
                        "in thoracolumbar trauma classification."
                    ),
                    "source": "local_evidence_corpus",
                }
            ],
            problem_row={
                "_stem": (
                    "Patient 1: Severe burst fracture of L2, no neurologic deficits. "
                    "Prioritize according to surgical indications."
                ),
                "_problem": {
                    "answer_type": "multipleChoice",
                    "category": "Biology/Medicine",
                    "raw_subject": "Medicine",
                },
            },
            query_row={
                "_option_text": "Patient 1",
                "_source_diagnostic_option_text": (
                    "Patient 1 Severe burst fracture of L2 no neurologic deficits"
                ),
            },
            query="Patient 1 severe burst fracture surgical indications",
        )

        self.assertEqual(detail["answer_bearing_diagnostics_status"], "evaluated")
        self.assertGreaterEqual(detail["answer_bearing_option_signal_count"], 1)
        self.assertGreaterEqual(detail["answer_bearing_directish_count"], 1)

    def test_source_diagnostics_include_medical_permutation_ordering(self):
        detail = prefetch._source_rows_answer_bearing_diagnostics(
            rows=[
                {
                    "title": "Thoracolumbar injury classification guideline",
                    "snippet": (
                        "TLICS thoracolumbar trauma classification assigns points "
                        "for morphology, neurologic status, and surgical treatment "
                        "indications."
                    ),
                    "source": "local_guideline",
                }
            ],
            problem_row={
                "_stem": (
                    "Patient 1: Severe burst fracture of L2, no neurologic deficits. "
                    "Patient 2: Compression fracture of L2 with mild traumatic "
                    "spondylolisthesis of L1, no neurologic deficits. "
                    "Patient 3: Split fracture of L2, with mildly disordered pelvic "
                    "functions. Prioritize according to surgical indications."
                ),
                "_problem": {
                    "answer_type": "multipleChoice",
                    "category": "Biology/Medicine",
                    "raw_subject": "Medicine",
                },
            },
            query_row={
                "_option_text": "Patient 3, Patient 2, Patient 1",
                "_source_diagnostic_option_text": (
                    "Patient 3 split fracture pelvic functions "
                    "Patient 2 compression fracture spondylolisthesis "
                    "Patient 1 severe burst fracture"
                ),
            },
            query="Patient 3 Patient 2 Patient 1 surgical indications",
        )

        self.assertEqual(
            detail["medical_guideline_permutation_ordering_status"],
            "activated",
        )
        self.assertTrue(detail["medical_guideline_permutation_ordering_candidate_exact"])
        serialized = json.dumps(detail)
        self.assertNotIn("mildly disordered pelvic", serialized)

    def test_source_diagnostics_include_fe_pair_binding_partial_not_direct(self):
        detail = prefetch._source_rows_answer_bearing_diagnostics(
            rows=[
                {
                    "title": "MossWinn paramagnetic hyperfine structure examples",
                    "snippet": (
                        "Iron is present in the high spin ferric form "
                        "( Fe 3+, S = 5/2 ). The hyperfine magnetic "
                        "interaction tensor is modeled."
                    ),
                    "source": "local_fulltext",
                }
            ],
            problem_row={
                "_stem": (
                    "Which combination has the largest hyperfine field in 57Fe "
                    "Mossbauer spectroscopy?"
                ),
                "_problem": {
                    "answer_type": "multipleChoice",
                    "category": "Chemistry",
                    "raw_subject": "Chemistry",
                },
            },
            query_row={
                "_option_text": "planar S = 5/2 Fe(III)",
                "_source_diagnostic_option_text": (
                    "planar S = 5/2 Fe(III) Fe 3+ ferric"
                ),
            },
            query="planar S 5 2 Fe III ferric hyperfine field",
        )

        self.assertEqual(detail["fe_hyperfine_pair_binding_status"], "evaluated")
        self.assertEqual(detail["fe_hyperfine_pair_binding_partial_row_count"], 1)
        self.assertEqual(detail["fe_hyperfine_pair_binding_direct_row_count"], 0)

    def test_candidate_specific_answer_bearing_queries_preserve_chemical_option_phrase(self):
        problem = {
            "id_hash": "pid",
            "answer_type": "multipleChoice",
            "category": "Biology/Medicine",
            "raw_subject": "Biochemistry",
        }
        option_text = "methyl (E)-4-oxo-4-(prop-2-yn-1-ylamino)but-2-enoate"

        phrases = prefetch._source_prefetch_focus_phrases(option_text, max_phrases=4)
        self.assertTrue(
            any(
                "prop-2-yn-1-ylamino" in phrase and "but-2-enoate" in phrase
                for phrase in phrases
            )
        )
        pairs = prefetch._candidate_specific_answer_bearing_source_prefetch_queries(
            stem=(
                "The lysate was clicked with cy5-azide and the light condition "
                "has a stronger fluorescent signal. What molecule leads to the "
                "fluorescent difference for the second probe in HEK 293T cells "
                "after 417 nm irradiation?"
            ),
            option_label="C",
            option_text=option_text,
            options={
                "A": "2-fluoro-7-methoxy-9H-thioxanthen-9-one",
                "B": "phenoxyl radical",
                "C": option_text,
                "D": "carbene",
                "E": "cy5 azide",
            },
            problem=problem,
            agent_plan={"stages": {}},
        )

        exact_queries = [
            query
            for kind, query in pairs
            if kind == "candidate_specific_answer_bearing_exact_option"
        ]
        anchor_queries = [
            query
            for kind, query in pairs
            if kind == "candidate_specific_answer_bearing_experiment_anchor"
        ]
        self.assertTrue(exact_queries)
        self.assertTrue(any("prop-2-yn-1-ylamino" in query for query in exact_queries))
        self.assertTrue(any("evidence" in query.lower() for query in exact_queries))
        self.assertTrue(anchor_queries)
        self.assertTrue(any("293T" in query or "417" in query for query in anchor_queries))

    def test_candidate_specific_numeric_threshold_queries_bind_unit_equivalent_value(self):
        problem = {
            "id_hash": "pid",
            "answer_type": "multipleChoice",
            "category": "Chemistry",
            "raw_subject": "Chemistry",
        }
        pairs = prefetch._candidate_specific_answer_bearing_source_prefetch_queries(
            stem=(
                "Using any method of synthesis, which temperature is the coldest "
                "at which xenon tetrafluoride can still be produced?"
            ),
            option_label="F",
            option_text="-78 C",
            options={
                "A": "25 C",
                "B": "0 C",
                "F": "-78 C",
                "J": "-120 C",
            },
            problem=problem,
            agent_plan={"stages": {}},
        )

        self.assertTrue(pairs)
        kinds = [kind for kind, _query in pairs]
        queries = [query for _kind, query in pairs]
        self.assertIn("candidate_specific_numeric_threshold_same_row_primary", kinds)
        self.assertIn("candidate_specific_numeric_threshold_same_row_unit_variant", kinds)
        self.assertIn("candidate_specific_numeric_threshold_same_row_relation", kinds)
        self.assertIn("candidate_specific_numeric_threshold_exact_value", kinds)
        self.assertTrue(
            any(
                ("195 K" in query or "195.15 K" in query)
                and (
                    "xenon" in query.lower()
                    or "tetrafluoride" in query.lower()
                    or "xef4" in query.lower()
                )
                and "temperature" in query.lower()
                for query in queries
            )
        )
        same_row_queries = [
            query.lower()
            for kind, query in pairs
            if str(kind).startswith("candidate_specific_numeric_threshold_same_row")
        ]
        self.assertTrue(
            any(
                "-78 c" in query
                and ("coldest" in query or "lowest" in query)
                and ("synthesis" in query or "produced" in query)
                for query in same_row_queries
            )
        )
        self.assertTrue(all("gold" not in query.lower() for query in queries))

    def test_candidate_specific_numeric_threshold_queries_use_entity_anchor_for_unitless_options(self):
        problem = {
            "id_hash": "pid",
            "answer_type": "multipleChoice",
            "category": "Biology/Medicine",
            "raw_subject": "Medicine",
        }
        pairs = prefetch._candidate_specific_answer_bearing_source_prefetch_queries(
            stem=(
                "According to the BI-RADS assessment, which is the minimum category "
                "at which focal breast lesions using ultrasound elastography should "
                "undergo biopsy?"
            ),
            option_label="D",
            option_text="4",
            options={
                "A": "1",
                "B": "2",
                "C": "3",
                "D": "4",
            },
            problem=problem,
            agent_plan={"stages": {}},
        )

        self.assertTrue(pairs)
        kinds = [kind for kind, _query in pairs]
        self.assertIn("candidate_specific_numeric_threshold_same_row_entity_anchor", kinds)
        self.assertIn("candidate_specific_numeric_threshold_biomedical_pubmed_anchor", kinds)
        biomedical_queries = [
            query.lower()
            for kind, query in pairs
            if kind == "candidate_specific_numeric_threshold_biomedical_pubmed_anchor"
        ]
        self.assertTrue(
            any(
                "4" in query
                and "bi-rads" in query
                and "biopsy" in query
                and "ultrasound elastography" in query
                for query in biomedical_queries
            )
        )
        entity_queries = [
            query.lower()
            for kind, query in pairs
            if kind == "candidate_specific_numeric_threshold_same_row_entity_anchor"
        ]
        self.assertTrue(
            any(
                "4" in query
                and (
                    "bi-rads assessment" in query
                    or "focal breast lesions" in query
                    or "ultrasound elastography" in query
                )
                and ("minimum" in query or "threshold" in query)
                for query in entity_queries
            )
        )
        self.assertTrue(all("gold" not in query.lower() for _kind, query in pairs))

    def test_problem_query_records_prefer_pubmed_for_biomedical_numeric_entity_anchor(self):
        problem = {
            "id_hash": "pid",
            "answer_type": "multipleChoice",
            "category": "Biology/Medicine",
            "raw_subject": "Medicine",
        }
        records, _summary = prefetch._problem_query_records(
            problem=problem,
            stem=(
                "According to the BI-RADS assessment, which is the minimum category "
                "at which focal breast lesions using ultrasound elastography should "
                "undergo biopsy?"
            ),
            options={
                "A": "1",
                "B": "2",
                "C": "3",
                "D": "4",
            },
            agent_plan={"stages": {}},
            max_options=4,
            max_queries_per_problem=12,
            max_queries_per_option=6,
            enable_candidate_specific_answer_bearing_queries=True,
        )

        entity_records = [
            record
            for record in records
            if record.get("query_kind")
            == "candidate_specific_numeric_threshold_same_row_entity_anchor"
        ]
        self.assertTrue(entity_records)
        self.assertEqual(
            entity_records[0].get("preferred_sources", [])[0],
            "pubmed_pmc_fulltext",
        )
        self.assertIn("pubmed", entity_records[0].get("preferred_sources", []))
        self.assertIn("semantic_scholar", entity_records[0].get("preferred_sources", []))

    def test_numeric_same_row_backfill_queries_bind_source_anchor_value_and_relation(self):
        queries = prefetch._numeric_same_row_backfill_queries_from_source_row(
            stem=(
                "Using any method of synthesis, which temperature is the coldest "
                "at which xenon tetrafluoride can still be produced?"
            ),
            option_text="400 C",
            problem={
                "answer_type": "multipleChoice",
                "category": "Chemistry",
                "raw_subject": "Chemistry",
            },
            source_row={
                "title": "XeF4 preparation note",
                "snippet": "The entry mentions xenon tetrafluoride near 400 C.",
                "source": "semantic_scholar",
                "url": "https://example.org/xef4-preparation?compound=XeF4",
            },
        )

        self.assertTrue(queries)
        kinds = [kind for kind, _query in queries]
        self.assertIn(
            "candidate_specific_numeric_threshold_same_row_source_url_backfill",
            kinds,
        )
        self.assertIn(
            "candidate_specific_numeric_threshold_same_row_source_title_backfill",
            kinds,
        )
        url_queries = [
            query
            for kind, query in queries
            if kind == "candidate_specific_numeric_threshold_same_row_source_url_backfill"
        ]
        self.assertEqual(len(url_queries), 1)
        self.assertTrue(url_queries[0].startswith("https://example.org/xef4-preparation?compound=XeF4"))
        self.assertIn("400 C", url_queries[0])
        self.assertTrue("coldest" in url_queries[0] or "lowest" in url_queries[0])
        joined = " ".join(query.lower() for _kind, query in queries)
        self.assertIn("400 c", joined)
        self.assertIn("xef4", joined)
        self.assertTrue("coldest" in joined or "lowest" in joined)
        self.assertTrue("synthesis" in joined or "produced" in joined)

    def test_numeric_same_row_backfill_skips_semantic_scholar_metadata_url(self):
        queries = prefetch._numeric_same_row_backfill_queries_from_source_row(
            stem=(
                "Using any method of synthesis, which temperature is the coldest "
                "at which xenon tetrafluoride can still be produced?"
            ),
            option_text="400 C",
            problem={
                "answer_type": "multipleChoice",
                "category": "Chemistry",
                "raw_subject": "Chemistry",
            },
            source_row={
                "title": "XeF4 preparation note",
                "snippet": "The entry mentions xenon tetrafluoride near 400 C.",
                "source": "semantic_scholar",
                "url": "https://www.semanticscholar.org/paper/example",
            },
        )

        self.assertFalse(
            any(
                kind == "candidate_specific_numeric_threshold_same_row_source_url_backfill"
                for kind, _query in queries
            )
        )

    def test_numeric_same_row_adaptive_backfill_plan_uses_value_match_cache_rows(self):
        query_hash = prefetch.stable_hash({"query": "xef4 400 c temperature"})
        option_hash = prefetch.stable_hash({"option_label": "B"})
        query_plan = [
            {
                "seed_offset": 70,
                "status": "planned",
                "_problem": {
                    "answer_type": "multipleChoice",
                    "category": "Chemistry",
                    "raw_subject": "Chemistry",
                },
                "_stem": (
                    "Using any method of synthesis, which temperature is the coldest "
                    "at which xenon tetrafluoride can still be produced?"
                ),
                "_options": {"B": "400 C"},
                "problem_id_hash": "problem-hash",
                "question_hash": "question-hash",
                "query_records": [
                    {
                        "query_hash": query_hash,
                        "query_kind": "candidate_specific_numeric_threshold_same_row_primary",
                        "option_hash": option_hash,
                        "option_label_hash": option_hash,
                        "option_text_hash": prefetch.stable_hash({"option_text": "400 C"}),
                        "option_choice": "B",
                        "_query": "xef4 400 c temperature",
                        "_option_label": "B",
                        "_option_text": "400 C",
                    }
                ],
            }
        ]
        source_records = [
            {
                "query_hash": query_hash,
                "query_kind": "candidate_specific_numeric_threshold_same_row_primary",
                "source": "semantic_scholar",
                "numeric_same_row_value_match_count": 1,
                "numeric_same_row_direct_count": 0,
            }
        ]
        cache_rows = [
            {
                "title": "XeF4 preparation note",
                "snippet": (
                    "The entry mentions xenon tetrafluoride can still be produced "
                    "near 400 C in a preparation experiment."
                ),
                "source": "semantic_scholar",
                "url": "https://example.org/xef4-preparation",
            }
        ]

        with patch.object(prefetch, "_evidence_source_cache_get", return_value=cache_rows) as cache_get:
            backfill_plan = prefetch._numeric_same_row_adaptive_backfill_query_plan(
                query_plan=query_plan,
                source_records=source_records,
                source_limit=2,
            )

        self.assertEqual(len(backfill_plan), 1)
        self.assertTrue(cache_get.call_args.kwargs["include_url"])
        records = backfill_plan[0]["query_records"]
        self.assertTrue(records)
        self.assertTrue(
            all(
                record["query_kind"].startswith(
                    "candidate_specific_numeric_threshold_same_row_source_"
                )
                for record in records
            )
        )
        self.assertTrue(all(record["parent_query_hash"] == query_hash for record in records))
        url_records = [
            record for record in records
            if record["query_kind"]
            == "candidate_specific_numeric_threshold_same_row_source_url_backfill"
        ]
        self.assertTrue(url_records)
        self.assertTrue(url_records[0]["_query"].startswith("https://example.org/xef4-preparation"))
        self.assertIn("400 C", url_records[0]["_query"])
        self.assertEqual(url_records[0]["allowed_sources"], ["answer_web_fulltext"])
        serialized_safe = json.dumps(prefetch._sanitize_problem_plan(backfill_plan[0]))
        self.assertNotIn("400 C", serialized_safe)
        self.assertNotIn("XeF4 preparation note", serialized_safe)

    def test_numeric_same_row_adaptive_backfill_skips_value_only_source_rows(self):
        query_hash = prefetch.stable_hash({"query": "xef4 400 c temperature"})
        option_hash = prefetch.stable_hash({"option_label": "B"})
        query_plan = [
            {
                "seed_offset": 70,
                "status": "planned",
                "_problem": {
                    "answer_type": "multipleChoice",
                    "category": "Chemistry",
                    "raw_subject": "Chemistry",
                },
                "_stem": (
                    "Using any method of synthesis, which temperature is the coldest "
                    "at which xenon tetrafluoride can still be produced?"
                ),
                "_options": {"B": "400 C"},
                "problem_id_hash": "problem-hash",
                "question_hash": "question-hash",
                "query_records": [
                    {
                        "query_hash": query_hash,
                        "query_kind": "candidate_specific_numeric_threshold_same_row_primary",
                        "option_hash": option_hash,
                        "option_label_hash": option_hash,
                        "option_text_hash": prefetch.stable_hash({"option_text": "400 C"}),
                        "option_choice": "B",
                        "_query": "xef4 400 c temperature",
                        "_option_label": "B",
                        "_option_text": "400 C",
                    }
                ],
            }
        ]
        source_records = [
            {
                "query_hash": query_hash,
                "query_kind": "candidate_specific_numeric_threshold_same_row_primary",
                "source": "semantic_scholar",
                "numeric_same_row_value_match_count": 1,
                "numeric_same_row_direct_count": 0,
            }
        ]
        cache_rows = [
            {
                "title": "Unrelated numeric table",
                "snippet": "The table lists a baseline value near 400 C.",
                "source": "semantic_scholar",
                "url": "https://example.org/unrelated-table",
            }
        ]

        with patch.object(prefetch, "_evidence_source_cache_get", return_value=cache_rows):
            backfill_plan = prefetch._numeric_same_row_adaptive_backfill_query_plan(
                query_plan=query_plan,
                source_records=source_records,
                source_limit=2,
            )

        self.assertEqual(backfill_plan, [])

    def test_numeric_same_row_adaptive_backfill_uses_directish_url_without_value_match(self):
        query_hash = prefetch.stable_hash({"query": "BI-RADS 4 biopsy"})
        option_hash = prefetch.stable_hash({"option_label": "D"})
        query_plan = [
            {
                "seed_offset": 70,
                "status": "planned",
                "_problem": {
                    "answer_type": "multipleChoice",
                    "category": "Biology/Medicine",
                    "raw_subject": "Medicine",
                },
                "_stem": (
                    "According to the BI-RADS assessment, which is the minimum category "
                    "at which focal breast lesions using ultrasound elastography should undergo biopsy?"
                ),
                "_options": {"D": "4"},
                "problem_id_hash": "problem-hash",
                "question_hash": "question-hash",
                "query_records": [
                    {
                        "query_hash": query_hash,
                        "query_kind": "candidate_specific_numeric_threshold_biomedical_pubmed_anchor",
                        "option_hash": option_hash,
                        "option_label_hash": option_hash,
                        "option_text_hash": prefetch.stable_hash({"option_text": "4"}),
                        "option_choice": "D",
                        "_query": "BI-RADS 4 biopsy ultrasound elastography",
                        "_option_label": "D",
                        "_option_text": "4",
                    }
                ],
            }
        ]
        source_records = [
            {
                "query_hash": query_hash,
                "query_kind": "candidate_specific_numeric_threshold_biomedical_pubmed_anchor",
                "source": "pubmed",
                "numeric_same_row_value_match_count": 0,
                "numeric_same_row_direct_count": 0,
                "answer_bearing_directish_count": 1,
            }
        ]
        cache_rows = [
            {
                "title": "Breast elastography guideline",
                "snippet": (
                    "BI-RADS assessment discusses focal breast lesions and biopsy "
                    "under ultrasound elastography."
                ),
                "source": "pubmed_abstract",
                "url": "https://doi.org/10.1234/birads-biopsy",
            }
        ]

        with (
            patch.object(prefetch, "_evidence_source_cache_get", return_value=cache_rows),
            patch.object(
                prefetch,
                "numeric_same_row_source_diagnostics",
                return_value={
                    "numeric_same_row_value_match_count": 0,
                    "numeric_same_row_direct_count": 0,
                },
            ),
            patch.object(
                prefetch,
                "_source_rows_answer_bearing_diagnostics",
                return_value={"answer_bearing_directish_count": 1},
            ),
        ):
            backfill_plan = prefetch._numeric_same_row_adaptive_backfill_query_plan(
                query_plan=query_plan,
                source_records=source_records,
                source_limit=2,
            )

        self.assertEqual(len(backfill_plan), 1)
        records = backfill_plan[0]["query_records"]
        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0]["query_kind"],
            "candidate_specific_numeric_threshold_same_row_source_url_backfill",
        )
        self.assertEqual(records[0]["allowed_sources"], ["answer_web_fulltext"])
        self.assertTrue(records[0]["_query"].startswith("https://doi.org/10.1234/birads-biopsy"))
        self.assertIn("4", records[0]["_query"])

    def test_numeric_same_row_adaptive_backfill_uses_doi_url_without_parent_directish(self):
        query_hash = prefetch.stable_hash({"query": "BI-RADS 4 biopsy"})
        option_hash = prefetch.stable_hash({"option_label": "D"})
        query_plan = [
            {
                "seed_offset": 70,
                "status": "planned",
                "_problem": {
                    "answer_type": "multipleChoice",
                    "category": "Biology/Medicine",
                    "raw_subject": "Medicine",
                },
                "_stem": (
                    "According to the BI-RADS assessment, which is the minimum category "
                    "at which focal breast lesions using ultrasound elastography should undergo biopsy?"
                ),
                "_options": {"D": "4"},
                "problem_id_hash": "problem-hash",
                "question_hash": "question-hash",
                "query_records": [
                    {
                        "query_hash": query_hash,
                        "query_kind": "candidate_specific_numeric_threshold_biomedical_pubmed_anchor",
                        "option_hash": option_hash,
                        "option_label_hash": option_hash,
                        "option_text_hash": prefetch.stable_hash({"option_text": "4"}),
                        "option_choice": "D",
                        "_query": "BI-RADS 4 biopsy ultrasound elastography",
                        "_option_label": "D",
                        "_option_text": "4",
                    }
                ],
            }
        ]
        source_records = [
            {
                "query_hash": query_hash,
                "query_kind": "candidate_specific_numeric_threshold_biomedical_pubmed_anchor",
                "source": "crossref",
                "numeric_same_row_value_match_count": 0,
                "numeric_same_row_direct_count": 0,
                "answer_bearing_directish_count": 0,
            }
        ]
        cache_rows = [
            {
                "title": "Breast elastography guideline",
                "snippet": "Journal of ultrasound medicine; 10.1234/birads-biopsy",
                "source": "crossref",
                "url": "https://doi.org/10.1234/birads-biopsy",
            }
        ]

        with (
            patch.object(prefetch, "_evidence_source_cache_get", return_value=cache_rows),
            patch.object(
                prefetch,
                "numeric_same_row_source_diagnostics",
                return_value={
                    "numeric_same_row_value_match_count": 0,
                    "numeric_same_row_direct_count": 0,
                },
            ),
            patch.object(
                prefetch,
                "_source_rows_answer_bearing_diagnostics",
                return_value={"answer_bearing_directish_count": 0},
            ),
        ):
            backfill_plan = prefetch._numeric_same_row_adaptive_backfill_query_plan(
                query_plan=query_plan,
                source_records=source_records,
                source_limit=2,
            )

        self.assertEqual(len(backfill_plan), 1)
        records = backfill_plan[0]["query_records"]
        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0]["query_kind"],
            "candidate_specific_numeric_threshold_same_row_source_url_backfill",
        )
        self.assertEqual(records[0]["allowed_sources"], ["answer_web_fulltext"])
        self.assertEqual(
            records[0]["source_url_backfill_reason"],
            "url_enrichment_without_directish",
        )
        self.assertTrue(records[0]["source_url_hash"])

    def test_problem_query_records_include_answer_bearing_pair_binding_queries_when_enabled(self):
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
            max_queries_per_problem=10,
            max_queries_per_option=5,
            enable_relation_query_planner=False,
            enable_answer_bearing_pair_binding_queries=True,
            relation_query_planner_model="gpt-5.4-mini",
        )

        kinds = [record["query_kind"] for record in records]
        self.assertIn("answer_bearing_pair_binding", kinds)
        safe = prefetch._sanitize_problem_plan({"problem_id_hash": "pid", "query_records": records})
        serialized = json.dumps(safe)
        self.assertIn("answer_bearing_pair_binding", serialized)
        self.assertNotIn("Alpha transporter alters sucrose uptake", serialized)
        self.assertNotIn("Beta transporter changes raffinose secretion", serialized)
        self.assertNotIn("_query", serialized)

    def test_problem_query_records_include_candidate_specific_answer_bearing_queries_when_enabled(self):
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
            stem=(
                "Which mechanism explains altered aphid feeding under controlled "
                "sucrose conditions?"
            ),
            options=options,
            agent_plan={"stages": {}},
            max_options=2,
            max_queries_per_problem=10,
            max_queries_per_option=5,
            enable_relation_query_planner=False,
            enable_candidate_specific_answer_bearing_queries=True,
            relation_query_planner_model="gpt-5.4-mini",
        )

        kinds = [record["query_kind"] for record in records]
        self.assertIn("candidate_specific_answer_bearing_witness", kinds)
        safe = prefetch._sanitize_problem_plan({"problem_id_hash": "pid", "query_records": records})
        serialized = json.dumps(safe)
        self.assertIn("candidate_specific_answer_bearing_witness", serialized)
        self.assertNotIn("Alpha transporter alters sucrose uptake", serialized)
        self.assertNotIn("controlled sucrose conditions", serialized)
        self.assertNotIn("_query", serialized)

    def test_problem_query_records_include_multiple_candidate_specific_variants_when_enabled(self):
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
            stem=(
                "Which mechanism explains altered aphid feeding under controlled "
                "sucrose conditions?"
            ),
            options=options,
            agent_plan={"stages": {}},
            max_options=1,
            max_queries_per_problem=6,
            max_queries_per_option=6,
            enable_relation_query_planner=False,
            enable_candidate_specific_answer_bearing_queries=True,
            relation_query_planner_model="gpt-5.4-mini",
        )

        candidate_specific_kinds = {
            record["query_kind"]
            for record in records
            if str(record.get("query_kind") or "").startswith("candidate_specific_answer_bearing")
        }

        self.assertIn("candidate_specific_answer_bearing_witness", candidate_specific_kinds)
        self.assertGreaterEqual(len(candidate_specific_kinds), 2)

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

    def test_balanced_prefetch_query_mix_prioritizes_pair_binding_under_backfill_pressure(self):
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
            answer_bearing_pair_binding_query_pairs=[
                ("answer_bearing_pair_binding", "alpha beta direct relation")
            ],
            answer_bearing_binding_query_pairs=[
                ("answer_bearing_relation_binding", "alpha direct relation")
            ],
            answer_web_queries=["answer web direct relation"],
            option_queries=["option evidence phrase"],
            claim_queries=["option claim answer phrase"],
            max_queries=5,
        )

        kinds = [kind for kind, _query in pairs]

        self.assertEqual(len(pairs), 5)
        self.assertIn("answer_bearing_pair_binding", kinds)
        self.assertIn("answer_bearing_relation_binding", kinds)
        self.assertLess(
            kinds.index("answer_bearing_pair_binding"),
            kinds.index("answer_bearing_relation_binding"),
        )

    def test_balanced_prefetch_query_mix_keeps_candidate_specific_variants_under_backfill_pressure(self):
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
            candidate_specific_answer_bearing_query_pairs=[
                ("candidate_specific_answer_bearing_witness", "alpha controlled sucrose evidence"),
                ("candidate_specific_answer_bearing_relation", "alpha altered aphid feeding"),
                ("candidate_specific_answer_bearing_required_term", "alpha controlled condition"),
            ],
            required_relation_completion_query_pairs=[
                ("answer_bearing_required_relation_completion", "alpha relation completion")
            ],
            answer_web_queries=["answer web direct relation"],
            option_queries=["option evidence phrase"],
            claim_queries=["option claim answer phrase"],
            max_queries=6,
        )

        kinds = [kind for kind, _query in pairs]

        self.assertEqual(len(pairs), 6)
        self.assertIn("candidate_specific_answer_bearing_witness", kinds)
        self.assertIn("candidate_specific_answer_bearing_relation", kinds)
        self.assertIn("candidate_specific_answer_bearing_required_term", kinds)
        self.assertLess(
            kinds.index("answer_bearing_required_relation_completion"),
            kinds.index("candidate_specific_answer_bearing_relation"),
        )
        self.assertLess(
            kinds.index("candidate_specific_answer_bearing_required_term"),
            kinds.index("candidate_specific_answer_bearing_witness"),
        )

    def test_live_budget_prioritizes_candidate_specific_answer_bearing_queries(self):
        jobs = []
        for index, kind in enumerate([
            "option_claim_deterministic_relation",
            "option_claim_local_relation_expansion",
            "answer_bearing_required_relation_completion",
            "candidate_specific_answer_bearing_witness",
            "answer_bearing_pair_binding",
            "candidate_specific_answer_bearing_witness",
        ]):
            jobs.append({
                "action": "fetch_candidate",
                "record": {
                    "query_kind": kind,
                    "query_hash": f"q{index}",
                    "option_hash": "option-a",
                    "source": "semantic_scholar",
                },
                "query_row": {
                    "query_kind": kind,
                    "query_hash": f"q{index}",
                    "option_hash": "option-a",
                },
                "source": "semantic_scholar",
                "query": f"query {index}",
            })

        budgeted = prefetch._apply_source_prefetch_live_budget(
            jobs=jobs,
            max_live_calls=2,
            budget_policy="round_robin_by_problem",
        )

        selected_kinds = [
            job["record"]["query_kind"]
            for job in budgeted
            if job["action"] == "fetch"
        ]
        self.assertEqual(
            selected_kinds,
            [
                "answer_bearing_required_relation_completion",
                "candidate_specific_answer_bearing_witness",
            ],
        )

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

    def test_legal_sources_are_supported_but_not_default(self):
        self.assertNotIn("lso_rules", prefetch.DEFAULT_SOURCES)
        self.assertNotIn("courtlistener", prefetch.DEFAULT_SOURCES)

        sources = prefetch._normalize_sources(["lso_rules", "courtlistener"])

        self.assertEqual(sources, ["lso_rules", "courtlistener"])

    def test_answer_web_fulltext_source_is_supported_but_not_default(self):
        self.assertNotIn("answer_web_fulltext", prefetch.DEFAULT_SOURCES)

        sources = prefetch._normalize_sources(["answer_web_fulltext"])

        self.assertEqual(sources, ["answer_web_fulltext"])

    def test_pubchem_source_is_supported_but_not_default(self):
        self.assertNotIn("pubchem", prefetch.DEFAULT_SOURCES)

        sources = prefetch._normalize_sources(["pubchem"])

        self.assertEqual(sources, ["pubchem"])

    def test_pubmed_source_is_supported_but_not_default(self):
        self.assertNotIn("pubmed", prefetch.DEFAULT_SOURCES)

        sources = prefetch._normalize_sources(["pubmed"])

        self.assertEqual(sources, ["pubmed"])

    def test_pubmed_pmc_fulltext_source_is_supported_but_not_default(self):
        self.assertNotIn("pubmed_pmc_fulltext", prefetch.DEFAULT_SOURCES)

        sources = prefetch._normalize_sources(["pubmed_pmc_fulltext"])

        self.assertEqual(sources, ["pubmed_pmc_fulltext"])

    def test_local_evidence_corpus_source_is_supported_but_not_default(self):
        self.assertNotIn("local_evidence_corpus", prefetch.DEFAULT_SOURCES)

        sources = prefetch._normalize_sources(["local_evidence_corpus"])

        self.assertEqual(sources, ["local_evidence_corpus"])

    def test_fetch_source_supports_legal_sources(self):
        with (
            patch.object(
                prefetch,
                "_ontario_lso_rules_search",
                return_value=[{"title": "Rules", "source": "lso_rules"}],
            ) as lso_search,
            patch.object(
                prefetch,
                "_courtlistener_search",
                return_value=[{"title": "Case", "source": "courtlistener"}],
            ) as court_search,
        ):
            lso_rows = prefetch._fetch_source(
                source="lso_rules",
                query="law firm adequate measures",
                limit=2,
                timeout=1.0,
            )
            court_rows = prefetch._fetch_source(
                source="courtlistener",
                query="former client conflict",
                limit=2,
                timeout=1.0,
            )

        self.assertEqual(lso_rows[0]["source"], "lso_rules")
        self.assertEqual(court_rows[0]["source"], "courtlistener")
        lso_search.assert_called_once()
        court_search.assert_called_once()

    def test_fetch_source_supports_answer_web_fulltext_source(self):
        with patch.object(
            prefetch,
            "_answer_bearing_web_fulltext_search",
            return_value=[{"title": "Fulltext", "source": "answer_web_fulltext"}],
        ) as search:
            rows = prefetch._fetch_source(
                source="answer_web_fulltext",
                query="direct target relation",
                limit=2,
                timeout=1.0,
            )

        self.assertEqual(rows[0]["source"], "answer_web_fulltext")
        search.assert_called_once()

    def test_fetch_source_supports_pubchem_source(self):
        with patch.object(
            prefetch,
            "_pubchem_search",
            return_value=[{"title": "Cy5 azide", "source": "pubchem"}],
        ) as search:
            rows = prefetch._fetch_source(
                source="pubchem",
                query="cy5 azide fluorescent",
                limit=2,
                timeout=1.0,
            )

        self.assertEqual(rows[0]["source"], "pubchem")
        search.assert_called_once()

    def test_fetch_source_supports_pubmed_source(self):
        with patch.object(
            prefetch,
            "_pubmed_search",
            return_value=[{"title": "Probe workflow", "source": "pubmed_abstract"}],
        ) as search:
            rows = prefetch._fetch_source(
                source="pubmed",
                query="chemical probe fluorescent lysate",
                limit=2,
                timeout=1.0,
            )

        self.assertEqual(rows[0]["source"], "pubmed_abstract")
        search.assert_called_once()

    def test_fetch_source_supports_pubmed_pmc_fulltext_source(self):
        with patch.object(
            prefetch,
            "_pubmed_pmc_fulltext_search",
            return_value=[{"title": "Full text", "source": "pubmed_pmc_fulltext"}],
        ) as search:
            rows = prefetch._fetch_source(
                source="pubmed_pmc_fulltext",
                query="BI-RADS 4 biopsy",
                limit=2,
                timeout=1.0,
            )

        self.assertEqual(rows[0]["source"], "pubmed_pmc_fulltext")
        search.assert_called_once()

    def test_fetch_source_supports_local_evidence_corpus_source(self):
        with patch.object(
            prefetch,
            "_local_evidence_corpus_search",
            return_value=[{"title": "Guideline", "source": "guideline"}],
        ) as search:
            rows = prefetch._fetch_source(
                source="local_evidence_corpus",
                query="BI-RADS 4 biopsy",
                limit=2,
                timeout=1.0,
                problem={"category": "Biology/Medicine", "raw_subject": "Medicine"},
            )

        self.assertEqual(rows[0]["source"], "guideline")
        search.assert_called_once()
        self.assertEqual(search.call_args.kwargs["problem"]["raw_subject"], "Medicine")

    def test_run_source_prefetch_executes_local_source_during_dry_run(self):
        query_plan = [
            {
                "problem_id_hash": "problem-hash",
                "seed_offset": 70,
                "_problem": {
                    "answer_type": "multipleChoice",
                    "category": "Biology/Medicine",
                    "raw_subject": "Medicine",
                },
                "_stem": "Which BI-RADS category should undergo biopsy?",
                "operator_family_tags": ["evidence-bearing relation"],
                "query_records": [
                    {
                        "query_hash": "query-hash",
                        "query_kind": "candidate_specific_numeric_threshold_biomedical_pubmed_anchor",
                        "option_hash": "option-hash",
                        "option_label_hash": "option-hash",
                        "option_text_hash": "option-text-hash",
                        "option_choice": "D",
                        "_query": "BI-RADS 4 biopsy ultrasound elastography",
                        "_option_label": "D",
                        "_option_text": "4",
                    }
                ],
            }
        ]

        with (
            patch.object(prefetch, "_cache_status", return_value="miss"),
            patch.object(
                prefetch,
                "_fetch_source",
                return_value=[{"title": "Guideline", "source": "guideline"}],
            ) as fetch_source,
        ):
            records = prefetch._run_source_prefetch(
                query_plan=query_plan,
                sources=["local_evidence_corpus"],
                source_limit=2,
                timeout=1.0,
                execute_live=False,
                max_live_calls=0,
                delay_sec=0.0,
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source"], "local_evidence_corpus")
        self.assertEqual(records[0]["status"], "fetched")
        self.assertEqual(records[0]["row_count"], 1)
        fetch_source.assert_called_once()
        self.assertEqual(fetch_source.call_args.kwargs["problem"]["raw_subject"], "Medicine")


if __name__ == "__main__":
    unittest.main()
