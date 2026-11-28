"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_01", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "824608a5e3cac92a542787b55e71a9975561f16ce4f5bcf73900106d511e8c99"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Panel-conditioned regression for scalar and trajectory readouts.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.linalg import cho_factor, cho_solve\\nfrom scipy.optimize import minimize\\nfrom scipy.spatial.distance import cdist\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _panel(panel, trim=0):\\n    u = _affine(panel[\'u\'])\\n    v = panel[\'v\']\\n    keep = np.arange(len(u))\\n    for _ in range(5 if trim else 0):\\n        c = np.linalg.lstsq(u[keep], v[keep], rcond=None)[0]\\n        keep = np.argsort(np.linalg.norm(u @ c - v, axis=1))[:len(u)-trim]\\n    return np.linalg.lstsq(u[keep], v[keep], rcond=None)[0]\\n\\n\\ndef _universal(k, kt, z, zt, y, noise):\\n    p, pt = _affine(z), _affine(zt)\\n    f = cho_factor(k + noise*np.eye(len(z)))\\n    ip, iy = cho_solve(f, p), cho_solve(f, y)\\n    c = np.linalg.solve(p.T @ ip, p.T @ iy)\\n    return pt @ c + kt @ (iy-ip @ c)\\n\\n\\ndef _scalar_kernel(x, xt, z, zt, y):\\n    def parts(a, b, u, v):\\n        distance = cdist(u, v, \'sqeuclidean\')\\n        raw = .4*np.exp(-.23*cdist(a, b, \'sqeuclidean\'))\\n        r = np.sqrt(.30*distance)\\n        return (\\n            raw + np.exp(-.13*distance) + .05*(1+u @ v.T/4)**2,\\n            raw + (1+r+r*r/3)*np.exp(-r),\\n            raw + 1/(1+.025*distance),\\n        )\\n\\n    train = parts(x, x, z, z)\\n    test = parts(xt, x, zt, z)\\n    return sum(weight*_universal(k, kt, z, zt, y, .003)\\n               for weight, k, kt in zip((.4, .3, .3), train, test))\\n\\n\\ndef _matern(z, zt, target, poly):\\n    p = _affine(z)\\n    n, dof = len(z), len(z)-p.shape[1]\\n    eye = np.eye(n)\\n    d = (z[:, None]-z[None, :])**2\\n    q = poly*(1+z @ z.T/4)**2\\n    ref = np.log(.13)\\n\\n    def obj(h):\\n        weights, noise = np.exp(h[:4]), np.exp(h[4])\\n        r = np.sqrt(5*(d @ weights))\\n        em = np.exp(-r)\\n        k = (1+r+r*r/3)*em + q\\n        f = cho_factor(k+noise*eye)\\n        ip = cho_solve(f, p)\\n        ff = cho_factor(p.T @ ip)\\n        v = cho_solve(f, target)\\n        v -= ip @ cho_solve(ff, p.T @ v)\\n        rss = target @ v\\n        penalty = h[:4]-ref\\n        loss = (dof*np.log(rss/dof) + 2*np.log(f[0].diagonal()).sum()\\n                + 2*np.log(ff[0].diagonal()).sum() + .35*(penalty @ penalty))\\n        g = cho_solve(f, eye)-ip @ cho_solve(ff, ip.T)-dof/rss*np.outer(v, v)\\n        slope = -5/6*(1+r)*em\\n        grad = np.einsum(\'ij,ijk,ij,k->k\', g, d, slope, weights)\\n        return loss, np.r_[grad+.7*penalty, noise*np.trace(g)]\\n\\n    h = minimize(obj, np.r_[np.full(4, ref), np.log(.003)], jac=True,\\n                 method=\'L-BFGS-B\', bounds=[(-5, 1.5)]*4+[(-12, -2)]).x\\n    weights, noise = np.exp(h[:4]), np.exp(h[4])\\n\\n    def kernel(a, b):\\n        r = np.sqrt(5*cdist(a*np.sqrt(weights), b*np.sqrt(weights), \'sqeuclidean\'))\\n        return (1+r+r*r/3)*np.exp(-r) + poly*(1+a @ b.T/4)**2\\n\\n    return _universal(kernel(z,z), kernel(zt,z), z, zt, target, noise)\\n\\n\\ndef _components(study):\\n    train = study[\'train\']\\n    x, xt = train[\'x\'], study[\'test\'][\'x\']\\n    a, b = _panel(study[\'panel_a\'], 3), _panel(study[\'panel_b\'])\\n\\n    def transform(u):\\n        return _affine(_affine(u) @ a) @ b\\n\\n    w, wt = transform(x), transform(xt)\\n    z, zt = (w-w.mean(0))/w.std(0), (wt-w.mean(0))/w.std(0)\\n    scalar = _scalar_kernel(x, xt, z, zt, train[\'y\'])\\n    traj = train[\'traj\']\\n    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]\\n    amps = traj @ modes.T\\n    trajectory = np.array([np.column_stack([_matern(z, zt, col, poly) for col in amps.T]) @ modes\\n                           for poly in (0., .05)])\\n    return scalar, trajectory\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    y_hat, trajectories = _components(study)\\n    state_pred = trajectories.mean(axis=0)\\n    terminal = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])}\\n"}'

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
