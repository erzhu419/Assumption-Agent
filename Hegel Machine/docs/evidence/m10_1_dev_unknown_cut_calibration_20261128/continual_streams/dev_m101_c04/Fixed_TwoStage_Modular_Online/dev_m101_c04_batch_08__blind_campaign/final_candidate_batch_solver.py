"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_07", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "ec6d91636cce9ef003696945907b42e31b352f94ba680be668b234dbf821ba0b", "node_01": "979cac7692cf6835f38b09ec689183029d8692b07212498893fc1e5d9352d6c5"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _calibrate(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    n = np.roll(u, -1, axis=1)\\n    designs = np.stack((np.ones_like(u), u, n, u*n, u**3), axis=-1)\\n    weights = np.ones(len(u))\\n    for _ in range(10):\\n        w = np.sqrt(weights)\\n        coef = np.array([np.linalg.lstsq(designs[:, j]*w[:, None], v[:, j]*w, rcond=None)[0] for j in range(u.shape[1])])\\n        r = np.linalg.norm(np.einsum(\'nck,ck->nc\', designs, coef)-v, axis=1)\\n        scale = max(np.median(r), 1e-8)\\n        weights = 1/(1+(r/(2*scale))**2)\\n    keep = np.argsort(r)[:-3]\\n    return np.array([np.linalg.lstsq(designs[keep, j], v[keep, j], rcond=None)[0] for j in range(u.shape[1])]).T\\n\\n\\ndef _inverse(x, c):\\n    bias, linear, cross, product, cubic = c\\n    def forward(z):\\n        n = np.roll(z, -1)\\n        return bias + linear*z + cross*n + product*z*n + cubic*z**3\\n    return np.array([least_squares(lambda z: forward(z)-row, (row-bias)/linear, gtol=1e-11).x for row in x])\\n\\n\\n\\n\\ndef _quadratic(z):\\n    i, j = np.triu_indices(z.shape[1])\\n    return np.c_[_affine(z), z[:, i] * z[:, j]]\\n\\n\\ndef _scalar_coordinate(d, e, y):\\n    # Identify the scalar index and its response curve using only training rows.\\n    base = np.linalg.lstsq(d, y, rcond=None)[0]\\n    best = None\\n    def fit(kind, init, size, test_kind):\\n        nonlocal best\\n        span = d[:, :size]\\n        ev = e[:, :size]\\n        def evaluate(p, D, S, jac=False):\\n            if kind == \'index\':\\n                q = D @ p[:-2]\\n                k = np.exp(p[-1])\\n                h = np.tanh(k*q)\\n                if jac:\\n                    g = p[-2]*(1-h*h)*k\\n                    return np.c_[D*(1+g[:, None]), h, g*q]\\n                return q+p[-2]*h\\n            q = S @ p[d.shape[1]:-1]\\n            if kind == \'sin\':\\n                h, g = np.sin(q), np.cos(q)\\n            else:\\n                h = np.tanh(q)\\n                g = 1-h*h\\n            if jac:\\n                return np.c_[D, S*(p[-1]*g[:, None]), h]\\n            return D@p[:d.shape[1]]+p[-1]*h\\n        lo, hi = np.full(len(init), -20.), np.full(len(init), 20.)\\n        if kind == \'index\':\\n            lo[-1], hi[-1] = -3, 2.8\\n        opt = least_squares(lambda p:evaluate(p,d,span)-y, np.clip(init,lo+1e-5,hi-1e-5), jac=lambda p:evaluate(p,d,span,True), bounds=(lo,hi), max_nfev=120, ftol=1e-8)\\n        err = np.mean(opt.fun**2)\\n        # Complexity correction favors a single quadratic index over a second index.\\n        quality = err * (1 + 2*len(init)/(len(y)-len(init)))\\n        if best is None or quality < best[0]:\\n            p = opt.x\\n            if kind == \'index\' and test_kind == \'latent\':\\n                z0,z1 = d@p[:-2],e@p[:-2]\\n            else:\\n                z0,z1 = evaluate(p,d,span),evaluate(p,e,ev)\\n            best = quality,z0,z1,kind,err\\n    for a in (.2, .6, -.25):\\n        for k in (.6, 1.7, 3.):\\n            fit(\'index\',np.r_[base/(1+a*.45),a,np.log(k)],d.shape[1],\'latent\')\\n    if best[-1] > 0.00014:\\n        for size in (5, d.shape[1]):\\n            for kind in (\'tanh\',\'sin\'):\\n                for scale in (.6,1.7):\\n                    fit(kind,np.r_[base*.8,scale*base[:size],.5],size,\'prediction\')\\n    return best[1:3]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    s = inputs[\'__input__\']\\n    train = s[\'train\']\\n    c = _calibrate(s[\'panel_a\'])\\n    panel = s[\'panel_b\']\\n    b = np.linalg.lstsq(_affine(panel[\'u\']), panel[\'v\'], rcond=None)[0]\\n    def design(x):\\n        z = np.linalg.solve(b[1:].T, (_inverse(x,c)-b[0]).T).T\\n        return _quadratic(z)\\n    d,e = design(train[\'x\']),design(s[\'test\'][\'x\'])\\n    a0,a1 = _scalar_coordinate(d,e,train[\'y\'])\\n    # The affine trajectory readout needs the three supported dynamic modes.\\n    traj = train[\'traj\']\\n    centered = traj - traj.mean(axis=0)\\n    _,_,vt = np.linalg.svd(centered, full_matrices=False)\\n    modes = centered @ vt[:3].T\\n    w = np.linalg.lstsq(d,modes,rcond=None)[0]\\n    return {\'z_tr\':np.c_[a0,d@w], \'z_te\':np.c_[a1,e@w], \'y\':train[\'y\'], \'traj\':traj}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(z):\\n    return np.column_stack((np.ones(len(z)), z))\\n\\n\\ndef _scalar_basis(z, affine):\\n    return np.column_stack((affine, z[:, 0] * z[:, 2], z[:, 1] * z[:, 3]))\\n\\n\\ndef _scalar_predict(basis, params):\\n    score = basis @ params[:-2]\\n    return score + params[-2] * np.tanh(params[-1] * score)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    data = inputs[\'node_00\']\\n    z_tr, z_te = data[\'z_tr\'], data[\'z_te\']\\n    tr, te = _affine(z_tr), _affine(z_te)\\n    state_pred = te @ np.linalg.lstsq(tr, data[\'traj\'], rcond=None)[0]\\n\\n    basis = _scalar_basis(z_tr, tr)\\n    y = data[\'y\']\\n    linear = np.linalg.lstsq(basis, y, rcond=None)[0]\\n    fit = least_squares(\\n        lambda p: _scalar_predict(basis, p) - y,\\n        np.r_[0.75 * linear, 0.45, 1.5],\\n        bounds=(np.r_[np.full(len(linear), -np.inf), 0., 0.1],\\n                np.r_[np.full(len(linear), np.inf), 2., 8.]),\\n    )\\n    y_hat = _scalar_predict(_scalar_basis(z_te, te), fit.x)\\n    endpoint = state_pred[:, -1]\\n    return {\\n        \'y_hat\': y_hat,\\n        \'state_pred\': state_pred,\\n        \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), endpoint.mean(), endpoint.std()]),\\n    }\\n"}'

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
