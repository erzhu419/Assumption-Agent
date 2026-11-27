from __future__ import annotations

import _verified_downstream as _dn
from upstream_transducer import calibrate_upstream


def solve_batch(item: dict) -> dict:
    study = calibrate_upstream(item)
    pulses = study.get('reference_tracer_pulses')
    if pulses:
        # The downstream wall fit uses measured pulses only.
        study['reference_tracer_pulses'] = [p for p in pulses if p['y_pulse_raw'] is not None]
    return _dn.solve_batch(study)
