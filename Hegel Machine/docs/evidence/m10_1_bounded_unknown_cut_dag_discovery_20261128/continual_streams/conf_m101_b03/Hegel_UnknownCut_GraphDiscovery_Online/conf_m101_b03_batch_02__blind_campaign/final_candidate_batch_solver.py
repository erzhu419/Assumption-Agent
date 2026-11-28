"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_02", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "a9b812d02e2170560a6da9d690998806077dffc4ce8923f4aa2b8cdf811100ff"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Calibrate the inputs and recover a dual-operator response and relaxation model.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.linalg import cho_factor, cho_solve\\nfrom scipy.optimize import least_squares\\nfrom scipy.spatial.distance import cdist\\n\\n\\ndef _fit(design, target):\\n    return np.linalg.lstsq(design, target, rcond=None)[0]\\n\\n\\ndef _calibration(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    i, j = np.triu_indices(len(u), 1)\\n    du, dv = u[i] - u[j], v[i] - v[j]\\n    slope = np.array([np.median(dv[du[:, k] != 0, k] / du[du[:, k] != 0, k])\\n                      for k in range(u.shape[1])])\\n    shift = np.median(v - u * slope, axis=0)\\n    error = np.linalg.norm(v - shift - u * slope, axis=1)\\n    keep = error <= 4 * np.median(error)\\n    design = np.stack((np.ones_like(u[keep]), u[keep]), axis=2)\\n    shift, slope = np.column_stack([_fit(design[:, k], v[keep, k])\\n                                    for k in range(u.shape[1])])\\n    return shift, slope\\n\\n\\ndef _panel_outputs(panel, z):\\n    u = panel[\'u\']\\n\\n    def products(x):\\n        return np.column_stack((np.ones(len(x)), x * np.roll(x, 1, axis=1)))\\n\\n    a = products(z) @ _fit(products(u), panel[\'v1\'])\\n    target = panel[\'v2\']\\n\\n    def harmonic(x, p):\\n        frequency, weight = p\\n        return np.column_stack((np.ones(len(x)), x,\\n                                x*x - 2*weight*np.sin(frequency*x/2)**2))\\n\\n    def residual(p):\\n        f = harmonic(u, p)\\n        return (f @ _fit(f, target) - target).ravel()\\n\\n    fits = [least_squares(residual, (w, -.2), bounds=((.5, -1.), (5., 1.)))\\n            for w in (1.5, 2.5, 3.5)]\\n    p = min(fits, key=lambda fit: fit.cost).x\\n    c = harmonic(z, p) @ _fit(harmonic(u, p), target)\\n    return a, c\\n\\n\\ndef _relaxation(times, traj):\\n    def basis(log_rates):\\n        return -np.expm1(-times[:, None] * np.exp(log_rates))\\n\\n    def residual(log_rates):\\n        h = basis(log_rates)\\n        return (h @ _fit(h, traj.T) - traj.T).ravel()\\n\\n    rates = least_squares(residual, np.log([.4, 2.]), bounds=(-5., 3.)).x\\n    h = basis(np.sort(rates))\\n    return h, _fit(h, traj.T).T\\n\\n\\ndef _slow_mode(design, design_test, c, c_test, amplitude):\\n    scale = np.sqrt(np.mean(c*c, axis=0))\\n    c, c_test = c / scale, c_test / scale\\n\\n    def kernel(x, y):\\n        r = np.sqrt(5 / x.shape[1]) * cdist(x, y) / 1.25\\n        return (1 + r + r*r/3) * np.exp(-r)\\n\\n    cov = kernel(c, c)\\n    cov.flat[::len(c)+1] += .003\\n    factor = cho_factor(cov)\\n    inv_design = cho_solve(factor, design)\\n    inv_amp = cho_solve(factor, amplitude)\\n    trend = _fit(design.T @ inv_design, design.T @ inv_amp)\\n    return design_test @ trend + kernel(c_test, c) @ (inv_amp - inv_design @ trend)\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    train = study[\'train\']\\n    n = len(train[\'x\'])\\n    x = np.concatenate((train[\'x\'], study[\'test\'][\'x\']))\\n    shift, scale = _calibration(study[\'panel_a\'])\\n    a, c = _panel_outputs(study[\'panel_c\'], (x - shift) / scale)\\n    base = np.column_stack((np.ones(len(x)), a, c))\\n    paired = a*c\\n    response = np.column_stack((base, paired.sum(axis=1)))\\n    y_hat = response[n:] @ _fit(response[:n], train[\'y\'])\\n\\n    design = np.column_stack((base, paired))\\n    h, amplitudes = _relaxation(study[\'eval_times\'], train[\'traj\'])\\n    fast = design[n:] @ _fit(design[:n], amplitudes[:, 1])\\n    slow = _slow_mode(design[:n], design[n:], c[:n], c[n:], amplitudes[:, 0])\\n    state_pred = np.column_stack((slow, fast)) @ h.T\\n    terminal = state_pred[:, -1]\\n    diagnostics = np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred, \'diagnostics_vec\': diagnostics}\\n"}'

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
