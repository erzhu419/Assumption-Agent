# Frozen finite research selection; ordinary Common actions only.
from pathlib import Path
import math
_FROZEN_PLAN = {'source_refs': [{'file': '/mnt/hegel-snapshot-cow/pr3_v7_case002_paid_analysis_action_20261027/run/case_002/pr3/training.json', 'json_path': ['method_results', 32]}], 'ordinary_method_path': 'shared/v8_case002_P0_ordinary.py'}
_CONDITION_SOURCE = None
_CONDITION_FEATURES = []

def _condition(memory):
    if (memory.get('method_condition_support') or {}).get('can_apply') is not True:
        return None
    features = memory.get('method_condition_features') or {}
    if any(name not in features or features[name] is None
           or not isinstance(features[name], (int, float))
           or not math.isfinite(features[name]) for name in _CONDITION_FEATURES):
        return None
    namespace = {}
    exec(_CONDITION_SOURCE, namespace)
    return bool(namespace['predict']({'features': features, 'history': []}))

def select_actions(public_spec, observations, memory):
    path = memory.get('ordinary_method_path')
    if not path:
        return {'stop': {'status': 'author_work_request', 'request': {
            'type': 'AuthorWorkRequest', 'reason': 'ordinary_method_path_required',
            'required': ['frozen ordinary method manifest binding']}}}
    namespace = {'__file__': path}
    exec(compile(Path(path).read_text(encoding='utf-8'), path, 'exec'), namespace)
    if 'select_research_segment' not in namespace:
        return {'stop': {'status': 'author_work_request', 'request': {
            'type': 'AuthorWorkRequest', 'reason': 'ordinary_method_selection_adapter_required',
            'required': ['current ordinary source with the registered finite selection consumer'],
            'ordinary_method_path': path}}}
    plan = memory.get('research_selection_plan', _FROZEN_PLAN)
    return namespace['select_research_segment'](public_spec, observations, memory,
        plan=plan, ordinary_select_actions=namespace['select_actions'],
        condition=_condition if _CONDITION_SOURCE is not None else None)
