"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_04", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_02": 1, "node_03": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"], "node_02": ["node_00"], "node_03": ["__input__", "node_01", "node_02"]}, "max_fanout": 2, "node_code_shas": {"node_00": "133e3d220d4240c40876fc9b4735fd01eeed02109ccbe9fff46941fe892cfb84", "node_01": "dc24099a88d9dcdb234ba8491e8945020b200eebfdd45f917fef305a07f14fb8", "node_02": "73c3148eddfec50bec669d9617b207385cd02c7331a7b8c40ebd03c0996ec3a8", "node_03": "a8d5622b3718656b728350b9dd13b2cf9c3b92923315467be85c7df0a5594486"}, "shared_fanout_nodes": ["node_00"], "sink_node_id": "node_03", "topo_order": ["node_00", "node_01", "node_02", "node_03"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _basis(x):\\n    \\"\\"\\"For each channel, use its own input and its cyclic successor.\\"\\"\\"\\n    z = np.roll(x, -1, axis=1)\\n    return np.stack((np.ones_like(x), x, z, x*z, x**3), axis=-1)\\n\\n\\ndef _calibrate(u, v):\\n    a = _basis(u)\\n\\n    def fit(rows, w):\\n        return np.stack([\\n            np.linalg.lstsq(a[rows, j]*w[:, None], v[rows, j]*w, rcond=None)[0]\\n            for j in range(u.shape[1])\\n        ])\\n\\n    rows = np.arange(len(u))\\n    w = np.ones(len(u))\\n    # Shared Cauchy weights screen contamination at the row level.\\n    for _ in range(20):\\n        c = fit(rows, w)\\n        r = np.linalg.norm(np.sum(a*c, axis=-1)-v, axis=1)\\n        scale = max(2*np.median(r), 1e-9)\\n        w = scale/np.hypot(scale, r)\\n    keep = np.argsort(r)[:-3]\\n    return fit(keep, np.ones(len(keep)))\\n\\n\\ndef _forward(x, c):\\n    return np.sum(_basis(x)*c, axis=-1)\\n\\n\\ndef _jacobian(x, c):\\n    j = np.arange(x.shape[1])\\n    k = (j+1) % x.shape[1]\\n    jac = np.zeros((len(x), len(j), len(j)))\\n    jac[:, j, j] = c[:, 1] + c[:, 3]*x[:, k] + 3*c[:, 4]*x**2\\n    jac[:, j, k] = c[:, 2] + c[:, 3]*x\\n    return jac\\n\\n\\ndef _invert(y, c):\\n    linear = _jacobian(np.zeros_like(y[:1]), c)[0]\\n    x = np.linalg.solve(linear, (y-c[:, 0]).T).T\\n    for _ in range(30):\\n        r = _forward(x, c)-y\\n        if np.max(np.abs(r)) < 1e-11:\\n            break\\n        step = np.linalg.solve(_jacobian(x, c), r[:, :, None])[:, :, 0]\\n        trial = x-step\\n        norm = np.linalg.norm(r, axis=1)\\n        for _ in range(12):\\n            bad = np.linalg.norm(_forward(trial, c)-y, axis=1) > norm\\n            if not np.any(bad):\\n                break\\n            step[bad] *= 0.5\\n            trial[bad] = x[bad]-step[bad]\\n        x = trial\\n    return x\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    a, c = study[\'panel_a\'], study[\'panel_c\']\\n    coef = _calibrate(a[\'u\'], a[\'v\'])\\n    measured = np.concatenate((study[\'train\'][\'x\'], study[\'test\'][\'x\']))\\n    return {\'u\': c[\'u\'], \'v1\': c[\'v1\'], \'v2\': c[\'v2\'], \'x\': _invert(measured, coef)}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import minimize_scalar\\n\\n\\ndef _basis(x, frequency):\\n    return np.column_stack([np.ones(len(x)), x, np.sin(frequency*x), np.sin(2*frequency*x)])\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    p = inputs[\'node_00\']\\n    u, v, x = p[\'u\'], p[\'v1\'], p[\'x\']\\n\\n    def fit(frequency):\\n        f = _basis(u, frequency)\\n        coef = np.linalg.lstsq(f, v, rcond=None)[0]\\n        return f, coef\\n\\n    def residual(frequency):\\n        f, coef = fit(frequency)\\n        return np.mean((f@coef-v)**2)\\n\\n    frequency = minimize_scalar(residual, bounds=(2.7, 3.8), method=\'bounded\').x\\n    _, coef = fit(frequency)\\n    return {\'b\': _basis(x, frequency)@coef}\\n", "node_02": "import numpy as np\\nfrom scipy.optimize import minimize_scalar\\n\\n\\ndef _basis(x, frequency):\\n    return np.column_stack([np.ones(len(x)), x, np.cos(frequency*x)])\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    p = inputs[\'node_00\']\\n    u, v, x = p[\'u\'], p[\'v2\'], p[\'x\']\\n\\n    def fit(frequency):\\n        f = _basis(u, frequency)\\n        coef = np.linalg.lstsq(f, v, rcond=None)[0]\\n        return f, coef\\n\\n    def residual(frequency):\\n        f, coef = fit(frequency)\\n        return np.mean((f@coef-v)**2)\\n\\n    frequency = minimize_scalar(residual, bounds=(1., 1.5), method=\'bounded\').x\\n    _, coef = fit(frequency)\\n    return {\'d\': _basis(x, frequency)@coef}\\n", "node_03": "import numpy as np\\nfrom itertools import combinations_with_replacement\\n\\n\\ndef _readout_basis(x):\\n    mixed = np.column_stack([x[:, i]*x[:, j] for i, j in\\n                             combinations_with_replacement(range(x.shape[1]), 2)])\\n    return np.column_stack((np.ones(len(x)), x, .4*mixed))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    train = inputs[\'__input__\'][\'train\']\\n    b = inputs[\'node_01\'][\'b\']\\n    d = inputs[\'node_02\'][\'d\']\\n    f = _readout_basis(np.column_stack((b, d)))\\n    n = len(train[\'x\'])\\n    a, xt = f[:n], f[n:]\\n    gram = a@a.T\\n    cross = xt@a.T\\n    eye = np.eye(n)\\n    yh = cross@np.linalg.solve(gram+.0003*eye, train[\'y\'])\\n    state = cross@np.linalg.solve(gram+.001*eye, train[\'traj\'])\\n    end = state[:, -1]\\n    return {\'y_hat\': yh, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([yh.mean(), yh.std(), end.mean(), end.std()])}\\n"}'

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
