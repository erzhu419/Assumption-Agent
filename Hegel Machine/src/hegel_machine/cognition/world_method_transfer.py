"""Install shared authored programs and their separate prior-world evidence."""

from copy import deepcopy


def install_method_bundle(agent, bundle, laboratory):
    """Use the ordinary file and method contracts without importing world state."""
    for path, source in bundle["files"].items():
        agent._execute("write", {"path": path, "code": source}, laboratory)
    agent.knowledge.setdefault("files", {}).update(deepcopy(bundle["files"]))
    agent.inherited_files.update(bundle["files"])
    methods = []
    for definition in bundle["methods"]:
        result = agent._execute("run", {**deepcopy(definition), "operator": "method"}, laboratory)
        methods.append(result["method"]["id"])
    return {"files": list(bundle["files"]), "methods": methods,
            "evidence_scope": "Prior-world archive; current observations and answers are acquired locally."}
