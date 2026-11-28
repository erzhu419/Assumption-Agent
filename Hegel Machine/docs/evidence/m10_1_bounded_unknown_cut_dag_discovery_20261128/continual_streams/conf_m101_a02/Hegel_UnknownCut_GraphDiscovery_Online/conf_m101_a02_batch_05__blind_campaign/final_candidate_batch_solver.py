"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_06", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"], "node_02": ["__input__", "node_00", "node_01"]}, "max_fanout": 2, "node_code_shas": {"node_00": "d1f03ee4a574a398c740556a996324a09b27b30739626bfd83e776808c99f179", "node_01": "521c45d1e61ba8bbbc39295253ed285fd295b08b0d595e50ff55180a91c8e5da", "node_02": "f1b6d9907f460759d45e73c9197290ee128c5a2b2af455b154db0f6b2e1afc12"}, "shared_fanout_nodes": ["node_00"], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _fit(design, values):\\n    return np.linalg.lstsq(design, values, rcond=None)[0]\\n\\n\\ndef _calibration(panel):\\n    coefficients = []\\n    for u, v in zip(panel[\'u\'].T, panel[\'v\'].T):\\n        design = np.column_stack((np.ones(len(u)), u))\\n        keep = np.arange(len(u))\\n        for _ in range(3):\\n            c = _fit(design[keep], v[keep])\\n            residual = np.abs(v - design @ c)\\n            keep = np.argsort(residual)[:3 * len(u) // 4]\\n        good = residual <= 8 * np.median(residual)\\n        coefficients.append(_fit(design[good], v[good]))\\n    return np.asarray(coefficients).T\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    offset, gain = _calibration(study[\'panel_a\'])\\n    return {\\n        \'cal_tr\': (study[\'train\'][\'x\'] - offset) / gain,\\n        \'cal_te\': (study[\'test\'][\'x\'] - offset) / gain,\\n        \'panel_b\': study[\'panel_b\'],\\n    }\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _basis(x, shape):\\n    squared = x * x\\n    return np.column_stack((np.ones(len(x)), x, x * (-np.expm1(-shape * squared))))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    upstream = inputs[\'node_00\']\\n    panel = upstream[\'panel_b\']\\n    u, v = panel[\'u\'], panel[\'v\']\\n\\n    def fit(shape):\\n        basis = _basis(u, shape)\\n        coefficients = np.linalg.lstsq(basis, v, rcond=None)[0]\\n        return basis @ coefficients - v, coefficients\\n\\n    shape = least_squares(lambda k: fit(k)[0].ravel(), np.full(u.shape[1], 0.8),\\n                          bounds=(0.005, 5.0)).x\\n    _, coefficients = fit(shape)\\n    return {key: _basis(upstream[source], shape) @ coefficients\\n            for key, source in ((\'z_tr\', \'cal_tr\'), (\'z_te\', \'cal_te\'))}\\n", "node_02": "import numpy as np\\n\\n\\ndef _linear(z):\\n    return np.column_stack((np.ones(len(z)), z))\\n\\n\\ndef _label_basis(x, z):\\n    # The label readout couples calibrated and mixed coordinates.\\n    difference = x[:, 2] - z[:, 3]\\n    return np.column_stack((\\n        _linear(z),\\n        z[:, 0] * z[:, 2],\\n        z[:, 1] * z[:, 3],\\n        x[:, 2],\\n        difference ** 3,\\n    ))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    train = inputs[\'__input__\'][\'train\']\\n    calibrated = inputs[\'node_00\']\\n    mixed = inputs[\'node_01\']\\n    z_tr, z_te = mixed[\'z_tr\'], mixed[\'z_te\']\\n    label_coef = np.linalg.lstsq(_label_basis(calibrated[\'cal_tr\'], z_tr),\\n                                train[\'y\'], rcond=None)[0]\\n    state_coef = np.linalg.lstsq(_linear(z_tr), train[\'traj\'], rcond=None)[0]\\n    y_hat = _label_basis(calibrated[\'cal_te\'], z_te) @ label_coef\\n    state_pred = _linear(z_te) @ state_coef\\n    terminal = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(),\\n                                         terminal.mean(), terminal.std()])}\\n"}'

def _canon_digest(obj) -> str:
    h = hashlib.sha256()
    def _feed(v):
        if v is None: h.update(b'N;')
        elif isinstance(v, bool): h.update(b'B1;' if v else b'B0;')
        elif isinstance(v, (int, np.integer)): h.update(f'I:{int(v)};'.encode())
        elif isinstance(v, (float, np.floating)):
            fv = float(v)
            if math.isnan(fv): h.update(b'F:NaN;')
            elif math.isinf(fv): h.update(b'F:Inf;')
            else: h.update(b'F:' + np.asarray([0.0 if fv == 0.0 else fv], dtype='<f8').tobytes() + b';')
        elif isinstance(v, str): h.update(f'S:{len(v)}:{v};'.encode())
        elif isinstance(v, np.ndarray):
            h.update(f'A:{v.dtype.kind}:{v.shape}:'.encode())
            if v.dtype.kind == 'f':
                a = np.asarray(v, dtype=np.float64)
                nm = np.isnan(a)
                cl = np.where(nm, 0.0, a)
                cl = np.where(cl == 0.0, 0.0, cl).astype('<f8')
                h.update(cl.tobytes('C'))
            else:
                h.update(np.asarray(v, dtype='<i8').tobytes('C'))
            h.update(b';')
        elif isinstance(v, dict):
            h.update(f'D:{len(v)}:'.encode())
            for k in sorted(v.keys(), key=str): _feed(str(k)); _feed(v[k])
            h.update(b';')
        elif isinstance(v, (list, tuple)):
            h.update(f'L:{len(v)}:'.encode())
            for it in v: _feed(it)
            h.update(b';')
        else:
            h.update(repr(v).encode())
    _feed(obj)
    return h.hexdigest()

def _load_node_fn(nid: str, src: str):
    ns = {'__name__': f'_hegel_node_{nid}'}
    exec(compile(src, f'<dag_node_{nid}>', 'exec'), ns)
    fn = ns.get('run_node') or ns.get('solve_batch')
    if not callable(fn):
        raise RuntimeError(f'DAG node {nid} missing callable run_node/solve_batch')
    return fn

_COMPILED_FNS = {nid: _load_node_fn(nid, src) for nid, src in json.loads(_NODE_SOURCES).items()}
_SPEC_OBJ = json.loads(_DAG_SPEC)

def solve_batch(study: dict) -> dict:
    outputs = {'__input__': deepcopy(study)}
    exec_counts = {nid: 0 for nid in _SPEC_OBJ['topo_order']}
    non_identity_checks = {}
    for nid in _SPEC_OBJ['topo_order']:
        srcs = _SPEC_OBJ['incoming'][nid]
        if len(_SPEC_OBJ['topo_order']) == 1 and srcs == ['__input__']:
            node_in = deepcopy(outputs['__input__'])
        else:
            node_in = {s: deepcopy(outputs[s]) for s in srcs}
        in_dig = _canon_digest(node_in)
        out_v = _COMPILED_FNS[nid](node_in)
        exec_counts[nid] += 1
        if exec_counts[nid] != 1:
            raise RuntimeError(f'DAG node {nid} executed {exec_counts[nid]} times != 1')
        if not isinstance(out_v, dict) or not out_v:
            raise RuntimeError(f'DAG node {nid} must return a non-empty dict')
        out_dig = _canon_digest(out_v)
        if nid != _SPEC_OBJ['sink_node_id']:
            # Anti-identity-shell check: intermediate node cannot merely echo its raw predecessor dict
            echoed = (out_dig == in_dig) or any(out_dig == _canon_digest(outputs[s]) for s in srcs)
            if echoed:
                raise RuntimeError(f'DAG node {nid} failed anti-identity-shell check')
            non_identity_checks[nid] = True
        outputs[nid] = out_v
    trace_fp = os.environ.get('HEGEL_M101_DAG_TRACE_PATH')
    if trace_fp:
        trace_obj = {
            'dag_id': _SPEC_OBJ['dag_id'],
            'active_node_count': len(_SPEC_OBJ['topo_order']),
            'depth': _SPEC_OBJ['depth'],
            'max_fanout': _SPEC_OBJ['max_fanout'],
            'shared_fanout_nodes': _SPEC_OBJ['shared_fanout_nodes'],
            'topo_order': _SPEC_OBJ['topo_order'],
            'exec_counts': exec_counts,
            'all_exec_count_eq_1': all(v == 1 for v in exec_counts.values()),
            'non_identity_checks': non_identity_checks,
            'node_code_shas': _SPEC_OBJ['node_code_shas'],
        }
        open(trace_fp, 'w', encoding='utf-8').write(json.dumps(trace_obj, sort_keys=True))
    return outputs[_SPEC_OBJ['sink_node_id']]
