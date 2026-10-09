"""Transfer must not inflate author counts or expose targets while selecting code."""

from copy import deepcopy

from scripts import run_python_recall_frozen_transfer as runner


def inputs():
    sources = runner.read(runner.previous.SOURCE_OUTCOMES)["sources"]
    authors = runner.read(runner.AUTHORS / "summary.json")
    programs, attempts = runner.published_programs(authors, runner.AUTHORS)
    return sources, authors, programs, attempts


def test_real_publications_are_deduplicated_and_failed_authors_retained():
    _, authors, programs, attempts = inputs()
    assert len(attempts) == 20
    assert sum(item["published"] for item in attempts) == 10
    assert len(programs) == 4
    assert sorted(len(item["authors"]) for item in programs) == [1, 1, 1, 7]
    assert all(item["program_id"] is None for item in attempts if not item["published"])
    assert [item["status"] for item in attempts] == [item["status"] for item in authors["authors"]]
    assert runner.previous.total_costs([item["costs"] for item in attempts]) == authors["new_author_costs"]
    for item in programs:
        assert all(runner.read(path)["method_source"] == item["source"] for path in item["published_results"])


def test_new_targets_preserve_budgets_and_separate_instrument_response_and_return():
    sources, _, programs, attempts = inputs()
    before = deepcopy(sources)
    plan = runner.protocol(sources, programs, attempts)
    original = runner.read(runner.ORIGIN / "protocol.json")
    cases = plan["transfer_cases"]
    assert len(cases) == 24 and len({item["id"] for item in cases}) == 24
    assert sources == before
    for key in ("limits", "prefix", "procedure_step_limit", "evaluation"):
        assert plan[key] == original[key]
    old_cases = list(original["practice_cases"].values()) + original["transfer_cases"]
    assert not {item["task"]["seed"] for item in cases} & {item["task"]["seed"] for item in old_cases}
    assert not {item["confirmation_seed"] for item in cases} & {item["confirmation_seed"] for item in old_cases}
    for source_index in (1, 2, 3):
        cohort = {item["kind"]: item for item in cases if item["source_index"] == source_index}
        assert set(cohort) == set(runner.KINDS)
        old = sources[source_index]["case"]["apparatus_seed"]
        assert cohort["response_only"]["task"]["apparatus_seed"] == old
        assert cohort["response_only"]["task"]["response_seed"] != old
        assert cohort["instrument_only"]["task"]["apparatus_seed"] != old
        assert cohort["instrument_only"]["task"]["response_seed"] == old
        assert cohort["both"]["task"]["instrument_mode"] == "recurrent"
        returned = cohort["old_return"]["task"]["calibration_change"]
        assert returned == {"after_rows": 96, "apparatus_seed": old, "response_seed": old,
                            "instrument_mode": "finite2"}
        assert cohort["late_response_change"]["task"]["calibration_change"]["after_rows"] == 167


def test_shared_executor_uses_real_paid_prefix_and_no_model_calls(tmp_path):
    sources, _, programs, attempts = inputs()
    plan = runner.protocol(sources, programs, attempts)
    # Exercise a previously used practice, leaving the new frozen targets unseen.
    original = runner.read(runner.ORIGIN / "protocol.json")
    spec = original["practice_cases"]["case_0"]
    source = sources[original["author_source_index"]]
    control = {"source": (runner.previous.BRIDGE / "historical_bridge.py").read_text()}
    payloads = [runner.train(spec, method, source, plan, tmp_path / str(index))
                for index, method in enumerate((programs[0], control))]
    assert payloads[0]["agent"].observations[:3] == payloads[1]["agent"].observations[:3]
    for payload in payloads:
        training = payload["training"]
        assert training["agent_arm"] == "code_react" and training["online_questions"] == []
        assert training["costs"]["model_calls"] == training["costs"]["reported_tokens"] == 0
        assert payload["prefix_costs"]["observation_cost"] == 72
        assert payload["research_cpu_seconds"] > 0
        scored = runner.previous.score_episode(payload, plan)
        assert scored["confirmed_and_correct"] and not scored["false_confirmation"]
