"""Execute bounded question expressions on predictions or actual paid records.

Readout binds record_index/step to a complete action and public row. Model
readouts use only that public binding; evidence readouts additionally verify it
against the retained record. Neither path supplies the other's answer. These
checks validate declared context, not the truth of arbitrary authored prose.
An explicit record_index={experiment_id:...,action_index:...} uses that plan's
actual purchased record supplied by Common; integer indices remain absolute.
Readout with finite_readouts and statistic_ref.kind vector/mean/
empirical_distribution reuses the scalar reader on each declared row. Vectors
keep their declared order; empirical distributions describe only those rows.
"""

from copy import deepcopy
from math import isfinite


class _Unresolved(Exception):
    def __init__(self, status, reason, **details):
        self.result = {'status': status, 'reason': reason, **details}


def _unknown(reason, **details):
    raise _Unresolved('scoped_unknown', reason, **details)


def _invalid(reason, **details):
    raise _Unresolved('invalid_experiment_binding', reason, **details)


def _number(value):
    return type(value) in (int, float) and isfinite(value)


def numeric_answer_distance(left, right):
    """Absolute scalar distance or declared-coordinate maximum for a vector.

    Distribution answers need their own noise/test contract; they have no
    implicit distance here.
    """
    if _number(left) and _number(right):
        return abs(left - right)
    if (isinstance(left, list) and isinstance(right, list) and left
            and len(left) == len(right) and all(_number(x) for x in left + right)):
        return max(abs(a - b) for a, b in zip(left, right))
    return None


def _difference(left, right):
    if _number(left) and _number(right):
        return left - right
    if (isinstance(left, list) and isinstance(right, list) and left
            and len(left) == len(right) and all(_number(x) for x in left + right)):
        return [a - b for a, b in zip(left, right)]
    _invalid('numeric_answers_with_matching_shape_required')


def _context_value(context):
    return context.get('context_ref', context) if isinstance(context, dict) else context


class _Evaluation:
    def __init__(self, observations, predictor, context, compute_budget, record_bindings=None):
        if compute_budget is not None and (type(compute_budget) is not int or compute_budget < 0):
            raise ValueError('compute_budget must be a nonnegative integer or None')
        self.observations = observations
        self.predictor = predictor
        self.record_bindings = record_bindings or {}
        self.context = context
        self.budget = compute_budget
        self.compute_cost = 0
        self.evidence = set()

    def _charge(self):
        if self.budget is not None and self.compute_cost >= self.budget:
            _unknown('unknown_due_to_incomplete_search', detail='compute_budget')
        self.compute_cost += 1

    def _context_checks(self, query):
        if 'context_ref' in query:
            if self.context is None:
                _unknown('unknown_due_to_missing_evidence', detail='context_ref')
            if query['context_ref'] != _context_value(self.context):
                _invalid('context_mismatch')
        context = self.context if isinstance(self.context, dict) else {}
        units = context.get('units', context.get('output_unit'))
        if 'units' in query and units is not None and query['units'] != units:
            _invalid('unit_mismatch')
        for premise in query.get('premises', []):
            reference = premise.get('ref') if isinstance(premise, dict) else premise
            value = premise if type(premise) is bool else context.get('premises', {}).get(reference)
            if value is False:
                _invalid('premise_not_satisfied', premise=reference)
            if value is not True:
                _unknown('premise_unknown', premise=reference)
        if 'mapping_ref' in query:
            mapping = context.get('mappings', {}).get(query['mapping_ref'])
            if mapping is False:
                _invalid('mapping_not_applicable', mapping_ref=query['mapping_ref'])
            if mapping is not True:
                _unknown('unknown_due_to_uncertain_mapping', mapping_ref=query['mapping_ref'])

    def _readout(self, query):
        record_index, step = query.get('record_index'), query.get('step')
        planned = isinstance(record_index, dict)
        if planned:
            experiment, action_index = record_index.get('experiment_id'), record_index.get('action_index')
            if (not isinstance(experiment, str) or not experiment
                    or type(action_index) is not int or action_index < 0):
                _invalid('invalid_record_or_step')
        if ((not planned and (type(record_index) is not int or record_index < 0))
                or type(step) is not int or step < 0):
            _invalid('invalid_record_or_step')
        binding = query.get('binding')
        if not isinstance(binding, dict) or not all(key in binding for key in ('action', 'features', 'history')):
            _unknown('unknown_due_to_uncertain_mapping', detail='complete_public_binding_required')
        if (not isinstance(binding['action'], dict) or not isinstance(binding['features'], dict)
                or not isinstance(binding['history'], list)):
            _invalid('invalid_public_binding')
        target = query.get('target', 'response')
        context = self.context if isinstance(self.context, dict) else {}
        expected_target = context.get('target', 'response')
        units = query.get('units', context.get('units', context.get('output_unit')))
        if self.predictor is not None:
            if target != expected_target:
                _invalid('target_mismatch')
            public = {key: deepcopy(binding[key]) for key in ('features', 'history')}
            self._charge()
            try:
                value = self.predictor(public)
            except (ValueError, TypeError, KeyError, IndexError, ArithmeticError) as error:
                _unknown('unknown_due_to_unavailable_observation', detail='predictor_execution_failed',
                         error=f'{type(error).__name__}: {error}')
        else:
            if planned:
                records = self.record_bindings.get(experiment, [])
                if action_index >= len(records):
                    _unknown('unknown_due_to_missing_evidence', record_index=record_index)
                record_index = records[action_index]
            if not isinstance(self.observations, (list, tuple)) or record_index >= len(self.observations):
                _unknown('unknown_due_to_missing_evidence', record_index=record_index)
            record = self.observations[record_index]
            if record.get('action') != binding['action']:
                _invalid('action_mismatch', record_index=record_index)
            if target != record.get('target', record.get('target_kind', 'response')):
                _invalid('target_mismatch', record_index=record_index)
            if 'context_ref' in record and record['context_ref'] != _context_value(self.context):
                _invalid('record_context_mismatch', record_index=record_index)
            actual_units = record.get('units', record.get('output_unit'))
            if actual_units is not None and units is not None and actual_units != units:
                _invalid('unit_mismatch', record_index=record_index)
            if units is None:
                units = actual_units
            rows = record.get('rows', [])
            if step >= len(rows):
                _unknown('unknown_due_to_missing_evidence', record_index=record_index, step=step)
            row = rows[step]
            if row.get('features') != binding['features'] or list(row.get('history', [])) != binding['history']:
                _invalid('public_row_mismatch', record_index=record_index, step=step)
            indices = record.get('row_indices', [])
            if step >= len(indices) or type(indices[step]) is not int or 'target' not in row:
                _unknown('unknown_due_to_missing_evidence', detail='paid_value_or_row_id',
                         record_index=record_index, step=step)
            if not _number(row['target']):
                _unknown('unknown_due_to_unavailable_observation', detail='nonfinite_readout')
            self._charge()
            value = row['target']
            self.evidence.add(indices[step])
        if not _number(value):
            _unknown('unknown_due_to_unavailable_observation', detail='nonfinite_readout')
        return {'value': value, 'units': units}

    def _finite_readout(self, query):
        cases, statistic = query['finite_readouts'], query.get('statistic_ref')
        kind = statistic.get('kind') if isinstance(statistic, dict) else None
        if kind not in {'vector', 'mean', 'empirical_distribution'}:
            _unknown('unknown_due_to_opaque_program_semantics', detail='readout_statistic_required')
        if not isinstance(cases, list) or not cases:
            _unknown('unknown_due_to_missing_evidence', detail='nonempty_finite_readouts_required')
        values, units = [], []
        for index, case in enumerate(cases):
            if (not isinstance(case, dict) or case.get('op') != 'Readout'
                    or 'finite_readouts' in case):
                _invalid('finite_readout_requires_scalar_readouts', case_index=index)
            try:
                answer = self.evaluate({**{key: query[key] for key in ('units', 'target') if key in query},
                                        **case})
            except _Unresolved as error:
                error.result.update(scope={'kind': 'finite_readouts', 'case_count': len(cases),
                    'answered_case_count': len(values)}, failed_case_index=index)
                raise
            values.append(answer['value'])
            units.append(answer.get('units'))
        if any(unit != units[0] for unit in units) or ('units' in query and query['units'] != units[0]):
            _invalid('unit_mismatch')
        scope = {'kind': 'finite_readouts', 'case_count': len(cases), 'answered_case_count': len(values)}
        result = {'units': units[0], 'scope': scope, 'statistic_ref': deepcopy(statistic)}
        if kind == 'vector':
            result['value'] = values
        elif kind == 'mean':
            result['value'] = sum(value / len(values) for value in values)
            if not _number(result['value']):
                _unknown('unknown_due_to_unavailable_observation', detail='nonfinite_statistic')
        else:
            counts = {}
            for value in values:
                counts[value] = counts.get(value, 0) + 1
            result.update(value={str(float(key)): count / len(values) for key, count in counts.items()},
                          units=None, support_units=units[0])
        return result

    def evaluate(self, query):
        if not isinstance(query, dict):
            _invalid('query_must_be_an_object')
        self._context_checks(query)
        operation = query.get('op')
        if operation == 'AllOnFiniteSet':
            cases = query.get('finite_cases')
            if not isinstance(cases, list) or not cases:
                _unknown('unknown_due_to_missing_evidence', detail='nonempty_finite_cases_required')
            results, unresolved, violated = [], [], []
            for index, predicate in enumerate(cases):
                try:
                    answer = self.evaluate(predicate)
                except _Unresolved as error:
                    if error.result['status'] != 'scoped_unknown':
                        raise
                    unresolved.append({'case_index': index, **error.result})
                    continue
                if answer.get('check_status') in {'satisfied', 'violated'}:
                    satisfied = answer['check_status'] == 'satisfied'
                elif type(answer.get('value')) is bool:
                    satisfied = answer['value']
                else:
                    _invalid('finite_case_requires_a_predicate', case_index=index)
                results.append({'case_index': index, 'satisfied': satisfied})
                if not satisfied:
                    violated.append(index)
            scope = {'kind': 'finite_cases', 'case_count': len(cases),
                     'answered_case_count': len(results), 'unresolved_case_count': len(unresolved)}
            details = {'scope': scope, 'case_results': results, 'unresolved_cases': unresolved}
            if unresolved and not violated:
                first = unresolved[0]
                _unknown(first['reason'], **details)
            value = not violated
            return {'value': value, 'units': None,
                    'check_status': 'satisfied' if value else 'violated',
                    'finite_status': ('predicted_' if self.predictor is not None else 'observed_')
                                     + ('true' if value else 'false'),
                    'counterexample_case_indices': violated, **details}
        if operation == 'Readout':
            return self._finite_readout(query) if 'finite_readouts' in query else self._readout(query)
        if operation == 'Difference':
            left = self.evaluate(query.get('left'))
            right = self.evaluate(query.get('right'))
            if left.get('units') != right.get('units'):
                _invalid('unit_mismatch')
            value = _difference(left['value'], right['value'])
            if numeric_answer_distance(value, value) is None:
                _unknown('unknown_due_to_unavailable_observation', detail='nonfinite_difference')
            return {'value': value, 'units': left.get('units')}
        if operation == 'RelationResidual':
            expected, atol = query.get('expected'), query.get('atol', 0.)
            relation = query.get('relation', 'eq')
            if (numeric_answer_distance(expected, expected) is None
                    or not _number(atol) or atol < 0 or relation not in {'eq', 'le', 'ge', 'lt', 'gt'}):
                _invalid('invalid_relation_contract')
            result = self.evaluate(query.get('query'))
            residual = _difference(result['value'], expected)
            if numeric_answer_distance(residual, residual) is None:
                _unknown('unknown_due_to_unavailable_observation', detail='nonfinite_residual')
            components = residual if isinstance(residual, list) else [residual]
            satisfied = all({'eq': abs(item) <= atol, 'le': item <= atol,
                             'ge': item >= -atol, 'lt': item < -atol,
                             'gt': item > atol}[relation] for item in components)
            return {'value': residual, 'units': result.get('units'),
                    'check_status': 'satisfied' if satisfied else 'violated',
                    'relation': relation, 'expected': expected, 'atol': atol}
        if operation == 'CheckApplication':
            if 'context_ref' not in query:
                _unknown('unknown_due_to_uncertain_mapping', detail='application_context_required')
            return self.evaluate(query.get('query'))
        _unknown('unknown_due_to_opaque_program_semantics', op=operation)


def _evaluate(query, observations, predictor, context, compute_budget, record_bindings=None):
    if isinstance(query, dict) and 'query_semantics' in query:
        if context is None:
            context = query.get('context_ref')
        query = query['query_semantics'].get('query_expression')
    evaluation = _Evaluation(observations, predictor, context, compute_budget, record_bindings)
    try:
        answer = evaluation.evaluate(query)
        result = {'status': 'answered', **answer}
    except _Unresolved as error:
        result = error.result
    return {**result, 'evidence': sorted(evaluation.evidence), 'compute_cost': evaluation.compute_cost}


def evaluate_on_model(query, predictor, *, context=None, compute_budget=None):
    """Evaluate declared public rows with predictor({'features': ..., 'history': ...})."""
    if not callable(predictor):
        raise TypeError('predictor must be callable on a public row')
    return _evaluate(query, None, predictor, context, compute_budget)


def evaluate_on_evidence(query, observations, *, context=None, compute_budget=None, record_bindings=None):
    """Evaluate only matching actual records; unknown results never use a model."""
    return _evaluate(query, observations, None, context, compute_budget, record_bindings)
