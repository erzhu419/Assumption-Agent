"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_05", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "15ba01afaa8ad9f90e4b5cfe87cee3e74e292026140fc12e522b1e930dd74bed"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Predict scalar responses and trajectories from calibrated panel interactions.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares, minimize_scalar\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _fit(x, y):\\n    return np.linalg.lstsq(x, y, rcond=1e-10)[0]\\n\\n\\ndef _calibrate(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    bases = [_affine(np.column_stack((x, z, x*z, x**3)))\\n             for x, z in zip(u.T, np.roll(u, -1, axis=1).T)]\\n    ids = np.arange(len(u))\\n    for _ in range(8):\\n        coef = np.column_stack([_fit(f[ids], v[ids, j]) for j, f in enumerate(bases)])\\n        err = np.linalg.norm(np.column_stack([f @ coef[:, j] for j, f in enumerate(bases)])-v, axis=1)\\n        new_ids = np.flatnonzero(err < max(.002, 4*np.median(err)))\\n        if np.array_equal(ids, new_ids):\\n            break\\n        ids = new_ids\\n    return coef\\n\\n\\ndef _invert(x, coef):\\n    c0, c1, c2, c3, c4 = coef\\n    def residual(z, observed):\\n        zn = np.roll(z, -1)\\n        return c0 + c1*z + c2*zn + c3*z*zn + c4*z**3 - observed\\n    def jac(z, observed):\\n        d = np.diag(c1 + c3*np.roll(z, -1) + 3*c4*z*z)\\n        d[np.arange(4), np.roll(np.arange(4), -1)] = c2+c3*z\\n        return d\\n    return np.array([least_squares(residual, (row-c0)/c1, jac=jac, args=(row,),\\n                                   gtol=1e-12, ftol=1e-12, xtol=1e-12).x for row in x])\\n\\n\\ndef _views(study, z):\\n    pb, pc = study[\'panel_b\'], study[\'panel_c\']\\n    b = _affine(z) @ _fit(_affine(pb[\'u\']), pb[\'v\'])\\n    u, v = pc[\'u\'], pc[\'v1\']\\n    def soft(x, k):\\n        w = k[0]*x + k[1]\\n        return _affine(w/(1+np.abs(w)) + k[2]*x)\\n    def res(k):\\n        f = soft(u, k)\\n        return (f @ _fit(f, v)-v).ravel()\\n    k = least_squares(res, [1.5, 0., 0.], bounds=([.15, -4, -4], [7., 4, 4]), max_nfev=100).x\\n    p = soft(z, k) @ _fit(soft(u, k), v)\\n    def cosine(x, freq):\\n        return _affine(np.cos(freq*x))\\n    def loss(freq):\\n        f = cosine(u, freq)\\n        return np.mean((f @ _fit(f, pc[\'v2\']) - pc[\'v2\'])**2)\\n    freq = minimize_scalar(loss, bounds=(2., 2.8), method=\'bounded\').x\\n    q = cosine(z, freq) @ _fit(cosine(u, freq), pc[\'v2\'])\\n    return b, p, q\\n\\n\\ndef _outer(u, v):\\n    return (u[:, :, None]*v[:, None, :]).reshape(len(u), -1)\\n\\n\\ndef _design(study, x):\\n    z = _invert(x, _calibrate(study[\'panel_a\']))\\n    b, p, q = _views(study, z)\\n    return np.column_stack((z, p, q, _outer(b, p), _outer(p, q)))\\n\\n\\ndef _predict(design, values, train, test, penalty):\\n    a = design[train]\\n    center = a.mean(axis=0)\\n    scale = a.std(axis=0)\\n    f = (a-center)/scale\\n    offset = values[train].mean(axis=0)\\n    gram = f.T @ f + penalty*np.eye(f.shape[1])\\n    w = np.linalg.solve(gram, f.T @ (values[train]-offset))\\n    return (design[test]-center)/scale @ w + offset\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    data = study[\'train\']\\n    n = len(data[\'x\'])\\n    design = _design(study, np.vstack((data[\'x\'], study[\'test\'][\'x\'])))\\n    y = _predict(design, data[\'y\'], slice(n), slice(n, None), .003)\\n    state = _predict(design, data[\'traj\'], slice(n), slice(n, None), .0001)\\n    terminal = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])}\\n"}'

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
