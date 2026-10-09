"""A source method archive must not become fresh-world measurements or answers."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.code_provider_context import prepare_code_context
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.world_method_transfer import install_method_bundle
from test_discoveryworld_agent import LIMITS, ScriptedProvider
from test_discoveryworld_method_reuse import METHOD_PATH, PRIOR_EXECUTION, SOURCE, ReferenceLab


def source_bundle():
    return {"files": {METHOD_PATH: SOURCE,
                      "shared/source_execution.json": json.dumps(PRIOR_EXECUTION)},
            "methods": [{"id": "reference", "path": METHOD_PATH, "max_steps": 8,
                         "parameters": {"target": 7, "perturb_label": "Increase by 1",
                                        "restore_label": "Decrease by 1"},
                         "provenance": "source-authored procedure",
                         "evidence_file": "shared/source_execution.json",
                         "applicability": "Rebind a current reference and reversible public menu controls."}]}


def test_both_arms_receive_the_same_archive_without_inheriting_local_evidence(tmp_path):
    bundle = source_bundle()
    original = deepcopy(bundle)
    views = []
    for arm in ("hegel", "code_archive"):
        agent = DiscoveryWorldAgent(ScriptedProvider([]), tmp_path / arm, arm=arm, limits=LIMITS)
        agent.lab = lab = ReferenceLab()
        result = install_method_bundle(agent, bundle, lab)
        assert result["methods"] == ["reference"]
        assert not agent.questions and not agent.observations and not agent.events
        assert not agent.method_results and not agent.program_executions and not lab.actions
        assert agent.costs["model_calls"] == 0
        assert agent._provenance(METHOD_PATH) == {
            "inherited_program": True, "inherited_source_unchanged": True}
        agent._save_state()
        context = prepare_code_context(agent.workspace, agent._request(), schema={}, instructions="")
        method = context["method_decision"]["methods"][0]
        assert method["prior_execution"] == PRIOR_EXECUTION
        assert context["files"][METHOD_PATH] == SOURCE
        views.append(method)
    assert views[0] == views[1]
    assert bundle == original


@pytest.mark.parametrize("arm", ["hegel", "code_archive"])
def test_imported_method_executes_against_fresh_target_parameters(tmp_path, arm):
    agent = DiscoveryWorldAgent(ScriptedProvider([]), tmp_path, arm=arm, limits=LIMITS)
    agent.lab = lab = ReferenceLab()
    install_method_bundle(agent, source_bundle(), lab)
    result = agent._execute("run", {"operator": "procedure", "method_id": "reference",
                                   "parameters": {"target": 9}}, lab)
    assert result["status"] == "method_stopped"
    assert lab.actions == [{"action": "TELEPORT_TO_OBJECT", "arg1": 9},
                           {"action": "TALK", "arg1": 9},
                           {"chosen_dialog_option_int": 4}, {"chosen_dialog_option_int": 2}]
    assert [record["index"] for record in agent.observations] == [0, 1, 2, 3]
    assert agent.methods[0]["parameters"]["target"] == 7
    assert not agent.questions
    agent._execute("write", {"path": METHOD_PATH, "code": SOURCE + "\n# Target-side revision\n"}, lab)
    assert agent._provenance(METHOD_PATH) == {
        "inherited_program": True, "inherited_source_unchanged": False}


def test_source_archive_does_not_satisfy_current_method_requirements(tmp_path):
    bundle = source_bundle()
    bundle["methods"][0]["requires"] = ["current_control_response"]
    agent = DiscoveryWorldAgent(ScriptedProvider([]), tmp_path, arm="hegel", limits=LIMITS)
    agent.lab = lab = ReferenceLab()
    install_method_bundle(agent, bundle, lab)
    result = agent._execute("run", {"operator": "procedure", "method_id": "reference"}, lab)
    assert result["error"] == "Method requirements need renewed evidence"
    assert not agent.questions and not agent.observations and not lab.actions


def test_fresh_task_runner_installs_the_shared_bundle_before_the_first_decision(tmp_path, monkeypatch):
    from hegel_machine.cognition import discoveryworld_lab
    from scripts import run_discoveryworld_pilot as pilot

    class FreshLab(ReferenceLab):
        def __init__(self, *args):
            super().__init__()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    providers = []

    def provider(*args, **kwargs):
        instance = ScriptedProvider([("finish", {"lesson": "No new experiment executed"})])
        providers.append(instance)
        return instance

    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(source_bundle()))
    monkeypatch.setattr(pilot, "CodexCodeProvider", provider)
    monkeypatch.setattr(discoveryworld_lab, "DiscoveryWorldLab", FreshLab)
    plan = {"model": {"requested_alias": "unused", "reasoning_effort": "medium"},
            "limits": LIMITS, "runtime": str(tmp_path), "method_bundle": str(bundle_path)}
    result = pilot.run_task(plan, {"id": "fresh"}, "code_archive", tmp_path / "output")
    assert result["method_bundle_installation"]["methods"] == ["reference"]
    assert result["research"]["questions"] == []
    assert providers[0].requests[0]["files"][METHOD_PATH] == SOURCE
    assert providers[0].requests[0]["evidence"]["observation_count"] == 0
