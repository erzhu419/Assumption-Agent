"""Resolve saved Python prediction columns without executing their predictors.

The columns are finite execution results, not scientific Programs or new
measurements. Public feature/history identity remains necessary at every read.
"""

from copy import deepcopy
from math import isfinite
from pathlib import Path

from .research_episodes import load_source_values


def resolve_forecast_references(theories, workspace):
    """Read each small source file once, attaching transient columns to copies.

    ``forecast_ref`` names file, json_path and the exact recorded model. File
    size/mtime are retained for the caller's existing cache invalidation.
    Missing or invalid sources remain UNKNOWN rather than model refutations.
    """
    resolved, payloads = deepcopy(theories), {}
    for theory in resolved:
        reference = theory.get('forecast_ref')
        if reference is None:
            continue
        theory.pop('_retained_forecast', None)
        theory.pop('forecast_resolution', None)
        theory['source_status'] = 'unavailable_retained_forecast'
        try:
            if (not isinstance(reference, dict) or not isinstance(reference.get('file'), str)
                    or not isinstance(reference.get('json_path'), list)
                    or not isinstance(reference.get('model'), str) or not reference['model']):
                raise ValueError('complete_forecast_reference_required')
            path = Path(reference['file'])
            path = path if path.is_absolute() else Path(workspace) / path
            key = str(path.resolve())
            if key not in payloads:
                try:
                    stat = path.stat()
                    version = {'file': key, 'stat_size': stat.st_size, 'stat_mtime_ns': stat.st_mtime_ns}
                    loaded, = load_source_values([{'file': key, 'json_path': []}], workspace)
                    payloads[key] = (loaded['value'], version, None)
                except (OSError, ValueError) as error:
                    payloads[key] = (None, {'file': key}, error)
            value, version, error = payloads[key]
            theory['forecast_source_version'] = deepcopy(version)
            if error is not None:
                raise ValueError('retained_forecast_file_unavailable') from error
            for component in reference['json_path']:
                value = value[component]
            if not isinstance(value, dict) or value.get('model') != reference['model']:
                raise ValueError('recorded_forecast_model_mismatch')
            if value.get('returncode', 0) != 0:
                raise ValueError('retained_forecast_execution_failed')
            inputs, predictions = value.get('prediction_inputs'), value.get('predictions')
            if (not isinstance(inputs, list) or not isinstance(predictions, list) or not inputs
                    or len(inputs) != len(predictions)):
                raise ValueError('complete_retained_forecast_columns_required')
            if any(not isinstance(row, dict) or not isinstance(row.get('features'), dict)
                   or not isinstance(row.get('history'), list) for row in inputs):
                raise ValueError('retained_forecast_public_rows_required')
            if any(type(value) not in (int, float) or not isfinite(value) for value in predictions):
                raise ValueError('finite_retained_predictions_required')
            theory['_retained_forecast'] = {'model': value['model'],
                'prediction_inputs': [{key: deepcopy(row[key]) for key in ('features', 'history')}
                                      for row in inputs], 'predictions': deepcopy(predictions)}
            theory['source_status'] = 'retained_forecast_ready'
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            theory['forecast_resolution'] = {'status': 'scoped_unknown', 'reason': str(error)}
    return resolved


def retained_forecast_predictor(table):
    """Return an exact public-row lookup; an unrecorded row stays UNKNOWN."""
    def predict(public):
        values = [prediction for row, prediction in zip(table['prediction_inputs'], table['predictions'])
                  if row['features'] == public['features'] and row['history'] == public['history']]
        if not values:
            raise ValueError('public_row_missing_from_retained_forecast')
        if any(value != values[0] for value in values[1:]):
            raise ValueError('conflicting_predictions_for_same_public_row')
        return values[0]

    return predict
