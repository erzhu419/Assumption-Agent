"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_01", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "0755bec10b4eb43fd407d43062cee07112ca4263eb5cd68c8f1e957b54436a30"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Calibrated kernel readouts for scalar response and time trajectory.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.linalg import cho_factor, cho_solve\\nfrom scipy.optimize import minimize\\nfrom scipy.spatial.distance import cdist\\n\\n\\ndef _linear(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _calibration(panel, robust=False):\\n    u = _linear(panel[\'u\'])\\n    v = panel[\'v\']\\n    keep = np.arange(len(u))\\n    if robust:\\n        for _ in range(8):\\n            c = np.linalg.lstsq(u[keep], v[keep], rcond=None)[0]\\n            err = np.linalg.norm(v - u @ c, axis=1)\\n            keep = np.argsort(err)[:len(u) - len(u)//5]\\n        keep = np.flatnonzero(err < 10*np.median(err))\\n    return np.linalg.lstsq(u[keep], v[keep], rcond=None)[0]\\n\\n\\ndef _kernel_predict(k, kt, target, noise):\\n    f = cho_factor(k + noise*np.eye(len(k)))\\n    return kt @ cho_solve(f, target)\\n\\n\\ndef _matern(z, zt, y):\\n    delta = (z[:, None] - z[None, :])**2\\n    dt = (zt[:, None] - z[None, :])**2\\n    p, pt = _linear(z), _linear(zt)\\n    eye = np.eye(len(z))\\n\\n    def kernel(d, theta):\\n        r = np.sqrt(5*(d @ np.exp(theta[:4])))\\n        return (1+r+r*r/3)*np.exp(-r)\\n\\n    def objective(theta, predict=False):\\n        f = cho_factor(kernel(delta, theta) + np.exp(theta[4])*eye)\\n        ip, iy = cho_solve(f, p), cho_solve(f, y)\\n        g = p.T @ ip\\n        beta = np.linalg.solve(g, p.T @ iy)\\n        alpha = iy - ip @ beta\\n        if predict:\\n            return pt @ beta + kernel(dt, theta) @ alpha\\n        return ((len(y)-p.shape[1])*np.log((y-p@beta) @ alpha)\\n                + 2*np.log(np.diag(f[0])).sum() + np.linalg.slogdet(g)[1]\\n                + 1.5*np.sum((theta[:4]+1.5)**2))\\n\\n    optimum = minimize(objective, [-1.5]*4+[-6.], method=\'L-BFGS-B\',\\n                       bounds=[(-5, 2)]*4+[(-12, -2)])\\n    return objective(optimum.x, predict=True)\\n\\n\\ndef _trajectory(x, xt, z, zt, train):\\n    scale = np.sqrt(np.mean(x*x, axis=0))\\n    xs, xst = x/scale, xt/scale\\n    def kernel(a, b, c, d):\\n        return np.exp(-.2*cdist(c, d, \'sqeuclidean\')) + (1+a@b.T/4)**2\\n    am = _kernel_predict(kernel(xs, xs, z, z), kernel(xst, xs, zt, z), train, .002)\\n    # The public trajectories are low rank; discard independent higher time modes.\\n    mean = train.mean(axis=0)\\n    modes = np.linalg.svd(train-mean, full_matrices=False)[2][:2]\\n    return mean + ((am-mean) @ modes.T) @ modes\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    train = study[\'train\']\\n    x, xt = train[\'x\'], study[\'test\'][\'x\']\\n    a = _calibration(study[\'panel_a\'], robust=True)\\n    b = _calibration(study[\'panel_b\'])\\n    xa, xat = _linear(x) @ a, _linear(xt) @ a\\n    xab, xabt = _linear(xa) @ b, _linear(xat) @ b\\n    def rbf(l, r):\\n        return np.exp(-.2*cdist(l, r, \'sqeuclidean\'))\\n    y = train[\'y\']\\n    y_hat = .55*_matern(xa, xat, y) + .45*_kernel_predict(rbf(xab, xab), rbf(xabt, xab), y, .002)\\n    ai, bi = np.linalg.inv(a[1:]), np.linalg.inv(b[1:])\\n    def inverse(z):\\n        return ((z-b[0]) @ bi-a[0]) @ ai\\n    state = _trajectory(x, xt, inverse(x), inverse(xt), train[\'traj\'])\\n    last = state[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}\\n"}'

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
