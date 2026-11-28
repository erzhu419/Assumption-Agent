"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_07_r00", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "c08f8bb3af6383db99c196c8beedeb1756239352085e9caa8e0b16897c8f9f26", "node_01": "e86a8adbe4c2879c667f11f9274d44c3e745a3e8e48675632a3cf0dfb686f162"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import minimize_scalar\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _fit(x, y):\\n    return np.linalg.lstsq(x, y, rcond=None)[0]\\n\\n\\ndef _align(panel):\\n    design, measured = _affine(panel[\'u\']), panel[\'v\']\\n    weights = np.ones(len(design))\\n    for _ in range(20):\\n        scale = np.sqrt(weights)[:, None]\\n        coeff = _fit(design * scale, measured * scale)\\n        distance = np.linalg.norm(measured - design @ coeff, axis=1)\\n        weights = np.minimum(1., 1.5 * np.median(distance) / np.maximum(distance, 1e-12))\\n    # Refit without the three documented corrupted calibration rows.\\n    inliers = np.argsort(distance)[:-3]\\n    return _fit(design[inliers], measured[inliers])\\n\\n\\ndef _mixer_basis(u, frequency):\\n    return np.column_stack((_affine(u), np.sin(frequency * u)))\\n\\n\\ndef _mixer(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n\\n    def loss(frequency):\\n        design = _mixer_basis(u, frequency)\\n        return np.sum((v - design @ _fit(design, v)) ** 2)\\n\\n    grid = np.linspace(0.5, 5., 19)\\n    i = min(range(len(grid)), key=lambda j: loss(grid[j]))\\n    freq = minimize_scalar(loss, bounds=(grid[max(0, i-1)], grid[min(len(grid)-1, i+1)]),\\n                           method=\'bounded\').x\\n    return freq, _fit(_mixer_basis(u, freq), v)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    alignment = _align(study[\'panel_a\'])\\n    freq, coeff = _mixer(study[\'panel_b\'])\\n\\n    def encode(x):\\n        u = np.linalg.solve(alignment[1:].T, (x - alignment[0]).T).T\\n        return _mixer_basis(u, freq) @ coeff\\n\\n    return {\'z_tr\': encode(study[\'train\'][\'x\']),\\n            \'z_te\': encode(study[\'test\'][\'x\'])}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(z):\\n    return np.column_stack((np.ones(len(z)), z))\\n\\n\\ndef _label_features(z):\\n    cross = (z[:, :2] * z[:, 2:]).sum(axis=1)\\n    return np.column_stack((z, cross))\\n\\n\\ndef _label(z, z_test, y):\\n    x, x_test = _label_features(z), _label_features(z_test)\\n    initial = np.linalg.lstsq(_affine(x), y, rcond=None)[0][1:]\\n\\n    def design(features, p):\\n        index = features @ p[:-1] + p[-1]\\n        return np.column_stack((np.ones(len(index)), index, np.tanh(index)))\\n\\n    def residual(p):\\n        basis = design(x, p)\\n        return basis @ np.linalg.lstsq(basis, y, rcond=None)[0] - y\\n\\n    fits = (least_squares(residual, np.r_[scale * initial, 0.0], max_nfev=100)\\n            for scale in (0.6, 1.0, 1.6))\\n    p = min(fits, key=lambda fit: fit.cost).x\\n    coef = np.linalg.lstsq(design(x, p), y, rcond=None)[0]\\n    return design(x_test, p) @ coef\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    train = inputs[\'__input__\'][\'train\']\\n    z, z_test = (inputs[\'node_00\'][k] for k in (\'z_tr\', \'z_te\'))\\n    y_hat = _label(z, z_test, train[\'y\'])\\n    coef = np.linalg.lstsq(_affine(z), train[\'traj\'], rcond=None)[0]\\n    state_pred = _affine(z_test) @ coef\\n    last = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}\\n"}'

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
