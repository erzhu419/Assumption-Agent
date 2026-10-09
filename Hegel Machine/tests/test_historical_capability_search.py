"""Historical lookup must connect a current problem to usable retained work."""

from pathlib import Path

from hegel_machine.cognition.historical_capabilities import historical_capabilities
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent


def test_refit_query_finds_preexisting_mechanisms_and_current_python_bridge():
    found = historical_capabilities(query="refit", compact=True)
    assert {"0.38", "0.79", "0.84"} <= {
        entry["version"] for entry in found["versions"]}
    bridge = next(route for route in found["reuse_routes"]
                  if route["id"] == "python_evidence_repair")
    assert {"0.38", "0.79", "0.84"} <= set(bridge["historical_versions"])
    assert "src/hegel_machine/cognition/python_current_evidence_method.py" in bridge["current_modules"]
    assert bridge["requires"] and bridge["remaining_gap"]
    chinese = historical_capabilities(query="证据范围")
    assert "python_evidence_repair" in {route["id"] for route in chinese["reuse_routes"]}


def test_query_preserves_version_and_group_filters_and_has_no_fallback_matches():
    found = historical_capabilities(query="REFIT", group="revision")
    assert {"0.38", "0.79"} <= {entry["version"] for entry in found["versions"]}
    assert all("revision" in entry["capability_groups"] for entry in found["versions"])
    assert "0.84" not in {entry["version"] for entry in found["versions"]}
    exact = historical_capabilities(version="v0.79.0", query="refit", group="revision")
    assert [entry["version"] for entry in exact["versions"]] == ["0.79"]
    assert all("0.79" in route["historical_versions"] for route in exact["reuse_routes"])
    missing = historical_capabilities(query="refit unknown_capability_not_implemented")
    assert missing["count"] == 0
    assert missing["versions"] == missing["reuse_routes"] == []


def test_compact_problem_lookup_excludes_unrelated_history_and_keeps_native_routes():
    found = historical_capabilities(query="refit", version="0.79", compact=True)
    assert [route["id"] for route in found["reuse_routes"]] == ["python_evidence_repair"]
    assert found["reuse_routes"][0]["historical_versions"] == ["0.79"]
    assert found["capability_groups"]
    assert all(set(group) == {"id", "title", "count"} and group["count"] == 1
               for group in found["capability_groups"])
    assert found["versions"][0]["native_commands"] == ["unified"]
    full = historical_capabilities(query="refit", version="0.79")
    assert any(len(route["historical_versions"]) > 1 for route in full["reuse_routes"])
    assert all("versions" in group for group in full["capability_groups"])


def test_compact_lookup_retains_withdrawal_and_replacement_instructions():
    found = historical_capabilities(version="0.174", compact=True)
    assert found["count"] == 1
    entry = found["versions"][0]
    assert entry["disposition"] == "reverted"
    assert entry["replacement_versions"] == ["0.176"]
    assert entry["adoption_note"]
    assert found["dispositions"] == {"reverted": 1}


def test_reuse_routes_resolve_to_real_versions_implementations_and_evidence():
    catalogue = historical_capabilities()
    versions = {entry["version"] for entry in catalogue["versions"]}
    routes = catalogue["reuse_routes"]
    assert routes and len({route["id"] for route in routes}) == len(routes)
    root = Path(__file__).resolve().parents[1]
    for route in routes:
        assert route["historical_versions"]
        assert set(route["historical_versions"]) <= versions, route["id"]
        assert route["current_modules"] and route["evidence"], route["id"]
        for path in route["current_modules"]:
            assert (root / path).is_file(), (route["id"], path)
        for evidence in route["evidence"]:
            assert (root / evidence["path"]).is_file(), (route["id"], evidence["path"])
            assert evidence["result"], route["id"]


def test_agent_lookup_forwards_problem_query_and_compact_selection(tmp_path):
    laboratory = TemporalInquiryLab(task_index=1, apparatus_seed=71031)
    agent = TemporalOperatorAgent(None, tmp_path, arm="code_react", source_task=False,
        config=InquiryLLMConfig(execution_seconds=30.))
    compact = agent._execute("run", {"operator": "legacy_capabilities",
        "query": "refit", "version": "0.79", "group": "revision"}, laboratory)
    assert compact["count"] == 1
    assert compact["versions"][0]["version"] == "0.79"
    assert "current_modules" not in compact["versions"][0]
    assert "python_evidence_repair" in {
        route["id"] for route in compact["reuse_routes"]}
    full = agent._execute("run", {"operator": "legacy_capabilities",
        "query": "refit", "version": "0.79", "compact": False}, laboratory)
    assert full["versions"][0]["current_modules"]
    missing = agent._execute("run", {"operator": "legacy_capabilities",
        "query": "unknown_capability_not_implemented"}, laboratory)
    assert missing["count"] == 0
    assert laboratory.costs["observation_cost"] == 0
    assert agent.costs["model_calls"] == 0
