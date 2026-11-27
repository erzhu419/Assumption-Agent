from __future__ import annotations
import _verified_downstream as _dn
import upstream_transducer as _up

def solve_batch(item: dict) -> dict:
    return _dn.solve_batch(_up.calibrate_upstream(item))
