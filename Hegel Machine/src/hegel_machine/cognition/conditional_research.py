"""Retain authored conditional relations in their measured input coordinates.

The old world-research ``bind`` contract supplies rows; the existing Python
evaluator and dependency loader supply fitting, retention and parent calls.
No input variable, scientific relation or fitting family is selected here.
"""

from copy import deepcopy
import json
from math import isfinite
from pathlib import Path
import sys

from .code_agent import _copy_model_context, python_model_context, retain_evaluated_python_model
from .observable_research import _score, _update_support, return_model_answer
from .online_inquiry_llm import _PLAN_RUNNER


CONDITIONAL_INTERFACE = """
Conditional components use the same ordinary Python evaluation and bindings:
evaluate {component:'chosen_id',view:'binding.py',model:'relation.py',code:'optional model source'}.
The view defines bind(public_spec,observations,memory), returning
{rows:[{features:{chosen_input:number,...},target:number,history:[],evidence:[paid_row_indices]}]}.
observations contains the complete purchased laboratory records, including paid
measurements; memory contains the component ID. Extract the measured inputs and
target yourself, citing the paid row_indices used for each row. Return all
applicable paid rows, omitting records without the needed measurements. This
permits a conditional relation before a control-only predictor exists.
fit(rows) receives these bound rows; predict(row) receives the same feature and
history coordinates without the target. The component ID names this relation,
independently of which measured quantity is its target. A successful evaluation
retains its executable predictor and freezes the binding and fitting context.
For another predictor of the same relation, evaluate {component:'chosen_id',model:'new.py'}:
the frozen view supplies both candidates' current paid rows. A changed view needs
a distinct component ID, because it changes the question being tested.
conditional_models contains the retained components. New purchases rerun the
frozen binding and test the frozen predictor if its bound rows changed. Paid fit
is not independent confirmation. Features and labels extracted by the view are
the author's interpretation of the cited observations.
Use a component's retained model path in the existing bindings argument. Your
parent constructs the component's input rows and calls it through bind_models.
Final response confirmation still supplies only public controls and history;
conditional measured inputs must be supplied by your own executable composition.
"""

_BIND_RUNNER = (_PLAN_RUNNER.replace("import contextlib,json,runpy,sys",
                                   "import contextlib,json,runpy,sys,os")
    .replace("code=runpy.run_path(sys.argv[1])",
             "os.chdir(sys.argv[2])\n    code=runpy.run_path(sys.argv[1])")
    .replace("code['select_actions']", "code['bind']"))


def _available(records):
    return sorted({index for record in records
                   for index in record.get('row_indices', [record['observation_index']])})


def _bind(agent, laboratory, component, view):
    context, inner = python_model_context(agent.workspace, view)
    code, stdout, stderr, timeout = agent._python(
        [sys.executable, '-c', _BIND_RUNNER, str(context / inner), str(context)],
        json.dumps({'spec': laboratory.public_spec(), 'observations': agent.observations,
                    'memory': {'component': component}}))
    if code or timeout:
        raise ValueError(f'Component view {view} failed: {stderr[-4000:]}')
    binding = json.loads(stdout)
    entries = binding['rows']
    available = set(_available(agent.observations))
    for row in entries:
        evidence = row.get('evidence', [])
        if not evidence or any(type(index) is not int or index not in available for index in evidence):
            raise ValueError('each conditional row must cite actual paid row indices')
        values = [*row['features'].values(), row['target'],
                  *(value for point in row.get('history', []) for value in point.values())]
        if not row['features'] or any(not isfinite(float(value)) for value in values):
            raise ValueError('conditional rows need finite features, history and targets')
    return entries


def _indices(entries):
    return sorted({index for row in entries for index in row['evidence']})


def _return(agent, component, question=None):
    return_model_answer(agent, component, question,
                        entry=agent.conditional_models[component], kind='executable_conditional')


def _refresh(agent, laboratory, component):
    entry = agent.conditional_models[component]
    available = _available(agent.observations)
    if available == entry['available_observation_indices']:
        return None
    try:
        entries = _bind(agent, laboratory, component, entry['view'])
        if entries == entry['bound_rows']:
            entry['available_observation_indices'] = available
            return None
        result = _score(agent, entry['model'], entries, refit=False)
    except ValueError as error:
        entry['support'] = 'needs_reassessment'
        result = {'error': str(error)}
    else:
        entry.update(bound_rows=deepcopy(entries), available_observation_indices=available)
        _update_support(entry, result, _indices(entries))
    _return(agent, component)
    return {'component': component, 'evaluation': result, 'retained': deepcopy(entry)}


def refresh_components(agent, laboratory):
    """Reapply each retained view to new paid evidence before using its support."""
    return [result for component in agent.conditional_models
            if (result := _refresh(agent, laboratory, component)) is not None]


def evaluate_component(agent, arguments, laboratory, question=None):
    component, model = arguments['component'], arguments.get('model', 'model.py')
    incumbent = agent.conditional_models.get(component)
    supplied_view = arguments.get('view')
    if incumbent is not None:
        if supplied_view is not None:
            source = (agent.workspace / supplied_view).read_text(encoding='utf-8')
            if supplied_view not in {incumbent['source_view'], incumbent['view']} or source != incumbent['view_source']:
                raise ValueError('a changed component view needs a distinct component ID')
        reassessment = _refresh(agent, laboratory, component)
        if incumbent['available_observation_indices'] != _available(agent.observations):
            return {'component': component, 'error': 'retained conditional view still needs reassessment',
                    'reassessment': reassessment}
        entries, view = incumbent['bound_rows'], incumbent['view']
    else:
        if supplied_view is None:
            raise ValueError('a new conditional component requires its authored view')
        view, reassessment = supplied_view, None
        entries = _bind(agent, laboratory, component, view)
    frozen = '_evaluated_model' in Path(model).parts
    result = (deepcopy(reassessment['evaluation'])
              if incumbent and model == incumbent['model'] and reassessment
              else _score(agent, model, entries, refit=not frozen))
    indices = _indices(entries)
    result.update(component=component, executed_path=model, observation_indices=indices)
    if reassessment:
        result['reassessment'] = reassessment
    if incumbent and model == incumbent['model']:
        _update_support(incumbent, result, indices)
    elif result.get('returncode') == 0 and (
            incumbent is None or incumbent.get('mse') is None or result['mse'] < incumbent['mse']):
        version = 0 if incumbent is None else incumbent['version'] + 1
        directory = Path('_observable_models') / '_conditional' / component / str(version) / '_evaluated_model'
        saved = retain_evaluated_python_model(agent.workspace, model, [], result['mse'],
            destination=agent.workspace / directory,
            training_observations=None if frozen else entries)
        training_indices = indices
        if frozen:
            context, _ = python_model_context(agent.workspace, model)
            support_path = context / 'conditional_support.json'
            support = json.loads(support_path.read_text()) if support_path.exists() else {}
            training_indices = (support.get('training_observation_indices')
                                if support.get('component') == component else None)
        if incumbent is None:
            # The ordinary model snapshot already contains its authored view
            # and that view's helper files. Existing snapshots remain frozen.
            if '_evaluated_model' in Path(view).parts:
                frozen_view = view
            elif frozen:
                # Reusing a previously retained predictor must not substitute
                # its old workspace for the newly authored binding's context.
                view_directory = directory.parent.parent / 'view' / '_evaluated_model'
                copied_view = _copy_model_context(agent.workspace, view, agent.workspace / view_directory)
                frozen_view = str(view_directory / copied_view)
            else:
                frozen_view = str(directory / view)
            view_source = (agent.workspace / view).read_text(encoding='utf-8')
            source_view = view
        else:
            frozen_view, view_source, source_view = (
                incumbent['view'], incumbent['view_source'], incumbent['source_view'])
        entry = {'component': component, 'version': version,
            'model': str(directory / saved['model']), 'source_model': model,
            'view': frozen_view, 'source_view': source_view, 'view_source': view_source,
            'bound_rows': deepcopy(entries), 'available_observation_indices': _available(agent.observations),
            'training_observation_indices': deepcopy(training_indices)}
        _update_support(entry, result, indices)
        agent.conditional_models[component] = entry
        (agent.workspace / directory / 'conditional_support.json').write_text(json.dumps(entry), encoding='utf-8')
    if component in agent.conditional_models:
        _return(agent, component, question)
        result['conditional_model'] = deepcopy(agent.conditional_models[component])
    return result
