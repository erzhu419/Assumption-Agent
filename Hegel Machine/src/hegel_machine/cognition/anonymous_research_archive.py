"""Rebind actual anonymous inquiry archives for shared later research.

The source directory is read only. A caller can materialize ``knowledge['files']``
once as its reference archive, then independently in every practice workspace.
Host-only source records identify the relative frozen model context for the
existing ``python_model_context_matches`` comparison.
"""

from copy import deepcopy
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

from .code_agent import python_model_context, _bound_model_contexts, _model_context_ignored


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


def resolve_workspace_archive(archive, training=None):
    """Materialize the existing workspace protocol for an independent consumer.

    Complete knowledge fields and source references retain their original scope.
    Only the executable file view is read back through Common's exporter.
    """
    from .common_archive_inquiry import CommonArchiveInquiryAgent

    archive = deepcopy(archive)
    if 'files' in archive or 'workspace' not in archive:
        return archive
    workspace = Path(archive['workspace']).resolve()
    if training is None:
        saved = workspace / 'training.json'
        if not saved.is_file():
            saved = workspace / 'research_state.json'
        training = json.loads(saved.read_text(encoding='utf-8')) if saved.is_file() else {}
    view = SimpleNamespace(workspace=workspace,
        files=dict.fromkeys(training.get('inherited_files', [])), knowledge=archive)
    manifest_paths = ['shared/knowledge.json', *[event['arguments']['manifest']
        for event in training.get('events', []) if event.get('arguments', {}).get('manifest')]]
    model_paths = [training['model']] if training.get('model') else []
    evaluated = training.get('evaluated_model')
    if evaluated:
        model_paths.extend([evaluated['model'], evaluated['source_model']])
    archive['files'] = CommonArchiveInquiryAgent._result_files(
        view, archive.get('manifest', {}), manifest_paths, model_paths)
    return archive


def _expand_archive_native_refs(archive):
    # Practice results reference their actual canonical bank. Resolve only that
    # saved view when an independent caller explicitly requests a full archive.
    workspace = Path(archive['workspace']).resolve()
    state = None
    for question in archive.get('question_traces', []):
        reference = question.get('native_condition_research')
        if isinstance(reference, dict) and reference.get('full_content_file') == 'research_state.json':
            if state is None:
                state = json.loads((workspace / 'research_state.json').read_text(encoding='utf-8'))
            rounds = state
            for key in reference['json_path']:
                rounds = rounds[key]
            question['native_condition_research'] = deepcopy(rounds[:reference['round_count']])


def prepare_frozen_source_archive(source_study, base_archive):
    """Add finite-native source contexts to one shared target archive.

    The original archive remains at its old paths. Source facts are references;
    only executable contexts and their local parameters become target files.
    The theory library is selected separately by the target registration.
    This preparation covers the finite actor's native exports, retained base
    predictors, local parameter/fitting views and declared bound model contexts;
    it is not an arbitrary Python filesystem-dependency exporter.
    """
    source_study = Path(source_study).resolve()
    if not isinstance(base_archive, dict):
        base_archive = json.loads(Path(base_archive).read_text(encoding='utf-8'))
    knowledge = resolve_workspace_archive(base_archive)
    files = knowledge.setdefault('files', {})
    original_files = dict(files)
    catalog = knowledge.setdefault('source_catalog', [])
    roles = knowledge.setdefault('research_roles', {})
    predictors = roles.setdefault('predictors', [])
    if roles.get('predictor') and roles['predictor'] not in predictors:
        predictors.append(roles['predictor'])
    for model in knowledge.get('finite_native_author', {}).get('predictor_paths', []):
        if model not in predictors:
            predictors.append(model)
    # Published archives retain catalogs inside their original execution roots.
    # A nested catalog's model paths remain relative to that catalog's parent.
    for name, content in original_files.items():
        if Path(name).name == 'source_catalog.json':
            for entry in json.loads(content):
                if entry.get('model'):
                    model = str(Path(name).parent / entry['model'])
                    if model not in predictors:
                        predictors.append(model)
    plan = json.loads((source_study / 'protocol.json').read_text(encoding='utf-8'))
    source_refs = []
    facts = {'observations.json', 'model_observations.json', 'laboratory_observations.json', 'paid_trace.json'}

    def context_files(context, *, fit_view=False):
        # The old common archive is already installed once at the target root.
        # Explicit bound children are collected separately in their own context.
        result = {}
        for directory, subdirs, names in os.walk(context):
            subdirs[:] = [name for name in subdirs if name not in {
                'archive', '_evaluated_model', '_experiment_models', '_target_work',
                '_submitted_models', 'reference_archive', '__pycache__'}]
            for name in names:
                path = Path(directory) / name
                if (path.is_file() and path.suffix in {'.py', '.json', '.md', '.txt'}
                        and name not in facts and not _model_context_ignored(path, context)):
                    result[str(path.relative_to(context))] = path.read_text(encoding='utf-8')
        if fit_view:
            fitting = next((context / name for name in ('model_observations.json', 'observations.json')
                            if (context / name).is_file()), None)
            if fitting is not None:
                result[fitting.name] = fitting.read_text(encoding='utf-8')
        return result

    def has_fit(path):
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'))
        except SyntaxError:
            return False  # Failed source code stays in the ordinary error path.
        return any((isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'fit')
                   or (isinstance(node, (ast.Import, ast.ImportFrom))
                       and any((item.asname or item.name) == 'fit' for item in node.names))
                   for node in tree.body)

    for case in plan['cases']:
        for arm in case.get('arms', plan['arms']):
            workspace = source_study / case['id'] / arm
            training_ref = {'file': str(workspace / 'training.json'), 'json_path': []}
            state_ref = {'file': str(workspace / 'research_state.json'), 'json_path': []}
            training = json.loads(Path(training_ref['file']).read_text(encoding='utf-8'))
            state = json.loads(Path(state_ref['file']).read_text(encoding='utf-8'))
            source_refs.append(training_ref)
            models = list(state.get('exported_candidate_paths', []))
            for question in state.get('questions', []):
                for round_ in question.get('native_condition_research', []):
                    models.extend(item['model'] for field in ('candidates', 'exported_models')
                                  for item in round_.get(field, []) if item.get('model'))
            models.extend(state.get('research_candidates', {}))
            if training.get('model'):
                models.append(training['model'])
            evaluated = training.get('evaluated_model')
            if evaluated:
                models.extend([evaluated['model'], evaluated['source_model']])
            prefix = Path('source_research') / case['id'] / arm
            transferred, mappings = [], []
            collected = set()

            def collect(context, destination, *, fit_view=False):
                key = context.resolve(), destination, fit_view
                if key not in collected:
                    collected.add(key)
                    for name, content in context_files(context, fit_view=fit_view).items():
                        files[str(destination / name)] = content

            for model in dict.fromkeys(models):
                if not (workspace / model).is_file():
                    raise FileNotFoundError(workspace / model)
                context, inner = python_model_context(workspace, model)
                fitting = has_fit(context / inner)
                view = context_files(context, fit_view=fitting)
                parts = Path(model).parts
                snapshots = [index for index, part in enumerate(parts) if part == '_evaluated_model']
                relative = Path(*parts[:snapshots[-1] + 1]) if snapshots else Path('.')
                base_view = {str(Path(name).relative_to(relative)): content
                    for name, content in original_files.items()
                    if Path(name).is_relative_to(relative)
                    and (Path(name).name not in facts or fitting and Path(name).name in view)
                    and not set(Path(name).relative_to(relative).parts[:-1]).intersection({
                        'archive', '_evaluated_model', '_experiment_models', '_target_work', '_submitted_models'})}
                if model in original_files and view == base_view:
                    mapped = model
                else:
                    # Retain the existing innermost snapshot boundary. A plain
                    # source-root model gets that same boundary explicitly.
                    destination = prefix / (relative if relative.parts else Path('_evaluated_model'))
                    collect(context, destination, fit_view=fitting)
                    for child, child_model in _bound_model_contexts(context, inner):
                        collect(child, destination / child.relative_to(context),
                                fit_view=has_fit(child / child_model))
                    mapped = str(destination / inner)
                transferred.append(mapped)
                mappings.append({'model': mapped, 'source_model': model,
                    'directory': str(python_model_context(Path('/'), mapped)[0].relative_to('/')),
                    'source_ref': {'file': str(context / inner)}})
                if mapped not in predictors:
                    predictors.append(mapped)
            final = next((m for m in mappings if m['source_model'] == training.get('model')), None)
            catalog.append({'id': f'source_research:{case["id"]}:{arm}',
                'directory': final['directory'] if final else str(prefix),
                'model': final['model'] if final else None,
                'predictors': transferred, 'status': training['status'],
                'source_ref': training_ref, 'state_ref': state_ref,
                'observations_ref': {'file': str(workspace / 'laboratory_observations.json'), 'json_path': []},
                'costs_ref': {**training_ref, 'json_path': ['costs']},
                'fresh_confirmation_ref': {**training_ref, 'json_path': ['fresh_confirmation']},
                'model_sources': mappings})
    files['source_catalog.json'] = json.dumps(catalog, indent=2)
    receipt = json.loads((source_study / 'source_aggregation.json').read_text(encoding='utf-8'))
    index = Path(receipt['index_path'])
    if index.is_file():
        knowledge['theory_index'] = {'file': str(index.resolve()), 'json_path': []}
    knowledge['frozen_source_refs'] = source_refs
    return knowledge


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
            archive = resolve_workspace_archive(archive, training)
            if 'workspace' in archive:
                _expand_archive_native_refs(archive)
            # The extra existing snapshot boundary gives a root-level model.py
            # the same fitting cwd it originally had. A retained nested snapshot
            # keeps its additional boundary and therefore its own fitting view.
            container = Path("archive") / identifier
            directory = container / "_evaluated_model"
            for name, content in archive.get('files', {}).items():
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
