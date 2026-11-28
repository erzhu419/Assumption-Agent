"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_08", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "eeb3ad46caf34b63012e0c259297b0f6221878510a68c41417a38a7a644e68d1"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Calibrated multi-output regression for the public study.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.linalg import cho_factor, cho_solve\\nfrom scipy.optimize import minimize_scalar\\nfrom scipy.spatial.distance import cdist\\n\\n\\ndef _fit(a, b):\\n    return np.linalg.lstsq(a, b, rcond=None)[0]\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _coordinates(study, raw):\\n    a = study[\'panel_a\']\\n    u, v = a[\'u\'], a[\'v\']\\n    good = np.linalg.norm(v - u, axis=1) < 3\\n    j = np.arange(raw.shape[1]); k = np.roll(j, -1)\\n\\n    def terms(x):\\n        return np.stack((np.ones_like(x), x, x[:, k], x*x[:, k], x**3), axis=-1)\\n\\n    d = terms(u[good])\\n    c = np.array([_fit(d[:, i], v[good, i]) for i in j])\\n    x = raw.copy()\\n    for _ in range(18):\\n        err = np.einsum(\'nij,ij->ni\', terms(x), c) - raw\\n        if np.max(np.abs(err), initial=0.) < 1e-11:\\n            break\\n        jac = np.zeros((len(x), len(j), len(j)))\\n        jac[:, j, j] = c[:, 1] + c[:, 3]*x[:, k] + 3*c[:, 4]*x*x\\n        jac[:, j, k] = c[:, 2] + c[:, 3]*x\\n        x -= np.linalg.solve(jac, err[..., None])[..., 0]\\n\\n    b = study[\'panel_b\']\\n    def harm(p, w):\\n        return np.column_stack((_affine(p), np.sin(w*p)))\\n    def b_loss(w):\\n        m = harm(b[\'u\'], w)\\n        return np.sum((m @ _fit(m, b[\'v\']) - b[\'v\'])**2)\\n    w = minimize_scalar(b_loss, bounds=(2., 3.6), method=\'bounded\').x\\n    m = harm(b[\'u\'], w)\\n    return x, harm(x, w) @ _fit(m, b[\'v\'])\\n\\n\\ndef _kernel_options(x, z, n):\\n    def distance(a):\\n        a = a / a[:n].std(axis=0)\\n        return cdist(a, a, \'sqeuclidean\') / a.shape[1]\\n\\n    dx, dz = distance(x), distance(z)\\n    ds = distance(np.sin(2.8*x))\\n    for bandwidth in (.24, .34, .48, .68, .96, 1.36, 1.92):\\n        scale = 2*bandwidth**2\\n        yield np.exp(-(.5*dx + .5*ds)/scale)\\n        yield np.exp(-(.5*dx + .25*dz + .25*ds)/scale)\\n\\n\\ndef _predict(x, z, y):\\n    n = len(y)\\n    h = _affine(z)\\n    design = h[:n]\\n    eye = np.eye(n)\\n    scores, predictions = [], []\\n    for k in _kernel_options(x, z, n):\\n        for ridge in (.001, .005, .02, .08, .32, 1.28):\\n            factor = cho_factor(k[:n, :n] + ridge*eye, check_finite=False)\\n            inverse = cho_solve(factor, eye, check_finite=False)\\n            g = inverse @ design\\n            trend = np.linalg.solve(design.T @ g, g.T)\\n            precision = inverse - g @ trend\\n            weights = precision @ y\\n            # The projected precision gives deleted-row residuals with the affine trend refitted.\\n            loo = np.abs(weights / np.diag(precision)[:, None])\\n            scores.append(loo[:, 0].mean() + 1.4*loo[:, 1:].mean() + .2*loo[:, -1].mean())\\n            predictions.append(k[n:, :n] @ weights + h[n:] @ (trend @ y))\\n    scores = np.array(scores)\\n    blend = np.exp(-(scores - scores.min())/.006)\\n    return np.einsum(\'m,mij->ij\', blend/blend.sum(), predictions)\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    train = study[\'train\']\\n    x, z = _coordinates(study, np.vstack((train[\'x\'], study[\'test\'][\'x\'])))\\n    p = _predict(x, z, np.column_stack((train[\'y\'], train[\'traj\'])))\\n    y_hat, state = p[:, 0], p[:, 1:]\\n    end = state[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])}\\n"}'

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
