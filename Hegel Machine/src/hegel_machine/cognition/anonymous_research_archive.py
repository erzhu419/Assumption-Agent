"""Rebind actual anonymous inquiry archives for shared later research.

The source directory is read only. A caller can materialize ``knowledge['files']``
once as its reference archive, then independently in every practice workspace.
Host-only source records identify the relative frozen model context for the
existing ``python_model_context_matches`` comparison.
"""

from copy import deepcopy
import json
from pathlib import Path

from .code_agent import python_model_context


ARCHIVE_INTERFACE = """source_catalog.json lists the supplied earlier investigations.
Each entry's directory preserves that author's original relative files and paid
fitting view. To run an earlier ordinary script, enter that directory first; do
not run it against unrelated files at the new workspace root. Evaluate a listed
predictor using its complete catalog model path. The _evaluated_model directory
boundaries preserve its original execution and fitting context, including nested
retained snapshots. These archives include actual prior successes and failures;
their presence does not establish current applicability. New observations and
new models belong to the current workspace, not inside the retained source.
"""


def _public_trace(value, private_roots):
    """Keep paid authored activity, omitting provider usage and host locations."""
    if isinstance(value, dict):
        return {key: _public_trace(item, private_roots) for key, item in value.items()
                if key not in {"usage", "workspace", "snapshot_workspace"}}
    if isinstance(value, list):
        return [_public_trace(item, private_roots) for item in value]
    if isinstance(value, str):
        for root in private_roots:
            value = value.replace(root, ".")
    return value


def load_research_archive(source_root):
    """Return common executable knowledge and separate evaluator source records.

    Registration follows the source protocol, never terminal quality or success.
    ``sources`` is host metadata and must not be inserted into an agent request.
    Every authored file remains byte-for-byte identical as a Unicode string.
    """
    source_root = Path(source_root).resolve()
    plan = json.loads((source_root / "protocol.json").read_text(encoding="utf-8"))
    files, catalog, sources = {}, [], []
    for case in plan["cases"]:
        for arm in plan["arms"]:
            identifier = f"source_{len(sources):03d}"
            workspace = source_root / case["id"] / arm
            training = json.loads((workspace / "training.json").read_text(encoding="utf-8"))
            archive_path = workspace / "archive.json"
            archive = (json.loads(archive_path.read_text(encoding="utf-8")) if archive_path.exists()
                       else deepcopy(training.get("outgoing_knowledge", {})))
            # The extra existing snapshot boundary gives a root-level model.py
            # the same fitting cwd it originally had. A retained nested snapshot
            # keeps its additional boundary and therefore its own fitting view.
            container = Path("archive") / identifier
            directory = container / "_evaluated_model"
            for name, content in archive.get("files", {}).items():
                files[str(directory / name)] = content
            original_model = training.get("model")
            model = str(directory / original_model) if original_model else None
            private_roots = sorted({str(workspace), str(training.get("workspace", workspace)),
                                    str(source_root)}, key=len, reverse=True)
            trace_path = str(container / "paid_trace.json")
            files[trace_path] = json.dumps({
                "events": _public_trace(training.get("events", []), private_roots),
            }, indent=2)
            confirmation = training.get("fresh_confirmation") or {}
            entry = {
                "id": identifier, "directory": str(directory), "model": model,
                "status": training["status"],
                "fresh_confirmation": {key: confirmation[key] for key in
                    ("passed", "mse", "rows", "mse_threshold") if key in confirmation},
                "lessons": _public_trace(deepcopy(archive.get("lessons", [])), private_roots),
                "question_traces": _public_trace(deepcopy(archive.get("question_traces", [])), private_roots),
                "paid_trace": trace_path,
            }
            catalog.append(entry)
            if model:
                context, inner_model = python_model_context(source_root, model)
                relative_context = str(context.relative_to(source_root))
            else:
                relative_context, inner_model = None, None
            sources.append({
                "id": identifier, "case": case["id"], "arm": arm,
                "status": training["status"], "workspace": str(workspace),
                "costs": deepcopy(training.get("costs", {})),
                "source_model": original_model, "directory": str(directory), "model": model,
                "model_context_directory": relative_context, "inner_model": inner_model,
                "fresh_confirmation_passed": confirmation.get("passed") is True,
            })
    files["source_catalog.json"] = json.dumps(catalog, indent=2)
    predictors = [entry["model"] for entry in catalog if entry["model"] is not None]
    roles = {"predictors": predictors}
    if predictors:
        roles["predictor"] = predictors[0]
    knowledge = {
        "files": files, "research_roles": roles, "source_catalog": catalog,
        "archive_instructions": ARCHIVE_INTERFACE,
        "lessons": [lesson for entry in catalog for lesson in entry["lessons"]],
    }
    return knowledge, sources
