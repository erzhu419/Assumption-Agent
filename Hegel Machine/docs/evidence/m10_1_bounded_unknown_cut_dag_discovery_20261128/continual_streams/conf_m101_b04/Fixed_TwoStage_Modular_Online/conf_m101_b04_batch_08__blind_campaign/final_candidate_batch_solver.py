"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_08", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "0cc91a7a5d4d8b417f678d7d4f857733bc02d3258d5271357037fcd12f4709a3", "node_01": "131bb339b34ff06a35aed7af39bda391b73f64874058604691e170831f2b62a5"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _linear(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _calibration(panel, robust=False):\\n    u = _linear(panel[\'u\'])\\n    v = panel[\'v\']\\n    keep = np.arange(len(u))\\n    if robust:\\n        for _ in range(8):\\n            c = np.linalg.lstsq(u[keep], v[keep], rcond=None)[0]\\n            err = np.linalg.norm(v - u @ c, axis=1)\\n            keep = np.argsort(err)[:len(u) - len(u)//5]\\n        keep = np.flatnonzero(err < 10*np.median(err))\\n    return np.linalg.lstsq(u[keep], v[keep], rcond=None)[0]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    x, xt = train[\'x\'], study[\'test\'][\'x\']\\n    a = _calibration(study[\'panel_a\'], robust=True)\\n    b = _calibration(study[\'panel_b\'])\\n    xa, xat = _linear(x) @ a, _linear(xt) @ a\\n    xab, xabt = _linear(xa) @ b, _linear(xat) @ b\\n    ai, bi = np.linalg.inv(a[1:]), np.linalg.inv(b[1:])\\n\\n    def inverse(z):\\n        return ((z-b[0]) @ bi-a[0]) @ ai\\n\\n    z, zt = inverse(x), inverse(xt)\\n    scale = np.sqrt(np.mean(x*x, axis=0))\\n    return {\'xa\': xa, \'xat\': xat, \'xab\': xab, \'xabt\': xabt,\\n            \'z\': z, \'zt\': zt, \'xs\': x/scale, \'xst\': xt/scale,\\n            \'y\': train[\'y\'], \'traj\': train[\'traj\']}\\n", "node_01": "from itertools import combinations, combinations_with_replacement\\n\\nimport numpy as np\\n\\n\\ndef _pair_basis(x, frequency):\\n    pair = np.column_stack([x[:, i] + sign * x[:, j]\\n                            for i, j in combinations(range(4), 2)\\n                            for sign in (-1, 1)])\\n    return np.column_stack((np.tanh(frequency * x),\\n                            np.tanh(frequency * pair), np.cos(frequency * pair)))\\n\\n\\ndef _basis(x, w, frequency):\\n    quadratic = [x[:, i] * x[:, j]\\n                 for i, j in combinations_with_replacement(range(4), 2)]\\n    return np.column_stack((np.ones(len(x)), x, *quadratic,\\n                            _pair_basis(x, frequency), _pair_basis(w, frequency)))\\n\\n\\ndef _predict(x, xt, w, wt, target, frequency, penalty):\\n    design = _basis(x, w, frequency)\\n    scale = np.maximum(design.std(axis=0), 0.1)\\n    design /= scale\\n    reg = np.full(design.shape[1], penalty)\\n    reg[:15] *= 0.05\\n    reg[0] = 0\\n    coef = np.linalg.solve(design.T @ design + np.diag(reg), design.T @ target)\\n    return (_basis(xt, wt, frequency) / scale) @ coef\\n\\n\\ndef _kernel_predict(x, xt, target, *, gaussian=False):\\n    dist = np.sum((x[:, None] - x[None, :]) ** 2, axis=2)\\n    cross_dist = np.sum((xt[:, None] - x[None, :]) ** 2, axis=2)\\n    scale = 1.3 * np.median(dist[np.triu_indices(len(x), 1)])\\n\\n    def kernel(d):\\n        q = d / scale\\n        return np.exp(-q / 2) if gaussian else (1 + q / 4) ** -2\\n\\n    base = np.column_stack((np.ones(len(x)), x)) if gaussian else np.ones((len(x), 1))\\n    test_base = np.column_stack((np.ones(len(xt)), xt)) if gaussian else np.ones((len(xt), 1))\\n    system = kernel(dist) + 0.0003 * np.eye(len(x))\\n    solved = np.linalg.solve(system, np.column_stack((base, target)))\\n    p, r = np.split(solved, [base.shape[1]], axis=1)\\n    trend = np.linalg.solve(base.T @ p, base.T @ r)\\n    pred = test_base @ trend + kernel(cross_dist) @ (r - p @ trend)\\n    return pred[:, 0] if target.ndim == 1 else pred\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    d = inputs[\'node_00\']\\n    scalar_basis = _predict(d[\'xa\'], d[\'xat\'], d[\'xab\'], d[\'xabt\'], d[\'y\'], 1.8, 0.1)\\n    scalar_kernel = _kernel_predict(d[\'xa\'], d[\'xat\'], d[\'y\'])\\n    y = 0.2 * scalar_basis + 0.8 * scalar_kernel\\n    state = _kernel_predict(d[\'z\'], d[\'zt\'], d[\'traj\'], gaussian=True)\\n    final = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), final.mean(), final.std()])}\\n"}'

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
