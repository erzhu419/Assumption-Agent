"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_05_r02", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "03da8748a312dbdd8e3cb11c1f9ac5c63e3654423393688504b5ba93fb3fd7cb", "node_01": "44d22e49ad7ad385cc6b94397fd29281389c91466211d1c906399cc28a0eede5"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _fit(x, y):\\n    return np.linalg.lstsq(x, y, rcond=None)[0]\\n\\n\\ndef _fit_panel_a(panel):\\n    x, v = _affine(panel[\'u\']), panel[\'v\']\\n    coef = _fit(x, v)\\n    for _ in range(12):\\n        error = np.linalg.norm(v - x @ coef, axis=1)\\n        scale = max(np.median(error), 1e-12)\\n        weight = 1.0 / np.hypot(1.0, error / scale)\\n        coef = _fit(x * weight[:, None], v * weight[:, None])\\n    error = np.linalg.norm(v - x @ coef, axis=1)\\n    keep = np.argsort(error)[:-3]\\n    return _fit(x[keep], v[keep])\\n\\n\\ndef _panel_b_design(x, params):\\n    amplitude, frequency = params\\n    return _affine(x - amplitude * np.sin(frequency * x))\\n\\n\\ndef _fit_panel_b(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n\\n    def profile(params):\\n        x = _panel_b_design(u, params)\\n        coef = _fit(x, v)\\n        return (x @ coef - v).ravel(), coef\\n\\n    params = np.array([0.4, 2.85])\\n    residual, coef = profile(params)\\n    delta = np.eye(2) * 1e-5\\n    for _ in range(30):\\n        jac = np.column_stack([(profile(params + d)[0] - profile(params - d)[0]) / 2e-5 for d in delta])\\n        step = _fit(jac, -residual)\\n        if np.linalg.norm(step) < 1e-8:\\n            break\\n        loss = residual @ residual\\n        for _ in range(12):\\n            trial = params + step\\n            error, trial_coef = profile(trial)\\n            if error @ error < loss:\\n                params, residual, coef = trial, error, trial_coef\\n                break\\n            step *= 0.5\\n        else:\\n            break\\n    return params, coef\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    a = _fit_panel_a(study[\'panel_a\'])\\n    params, b = _fit_panel_b(study[\'panel_b\'])\\n\\n    def calibrate(x):\\n        u = np.linalg.solve(a[1:].T, (x - a[0]).T).T\\n        return _panel_b_design(u, params) @ b\\n\\n    train, test = study[\'train\'], study[\'test\']\\n    return {\\n        \'z\': calibrate(train[\'x\']),\\n        \'z_test\': calibrate(test[\'x\']),\\n        \'y\': train[\'y\'].copy(),\\n        \'traj\': train[\'traj\'].copy(),\\n    }\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(z):\\n    return np.column_stack((np.ones(len(z)), z))\\n\\n\\ndef _scalar_basis(z, affine):\\n    return np.column_stack((affine, z[:, 0] * z[:, 2], z[:, 1] * z[:, 3]))\\n\\n\\ndef _scalar_design(basis, direction):\\n    return np.column_stack((basis, np.tanh(basis @ direction)))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    data = inputs[\'node_00\']\\n    z, z_test = data[\'z\'], data[\'z_test\']\\n    y = np.asarray(data[\'y\']).reshape(-1)\\n    train, test = _affine(z), _affine(z_test)\\n    basis = _scalar_basis(z, train)\\n\\n    def fit_scalar(direction):\\n        design = _scalar_design(basis, direction)\\n        return design, np.linalg.lstsq(design, y, rcond=None)[0]\\n\\n    def residual(direction):\\n        design, weights = fit_scalar(direction)\\n        return design @ weights - y\\n\\n    start = [0., 1., 1., 1., -1., 0., 0.]\\n    direction = least_squares(residual, start, max_nfev=150).x\\n    _, weights = fit_scalar(direction)\\n    y_hat = _scalar_design(_scalar_basis(z_test, test), direction) @ weights\\n    state_pred = test @ np.linalg.lstsq(train, data[\'traj\'], rcond=None)[0]\\n    last = state_pred[:, -1]\\n    diagnostics = np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred, \'diagnostics_vec\': diagnostics}\\n"}'

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
