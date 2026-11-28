"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_07", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "465985c4b52b60aa78c56aa211818880d50030c4248fa1232ba18d803a5ab3f6", "node_01": "44d22e49ad7ad385cc6b94397fd29281389c91466211d1c906399cc28a0eede5"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _fit_panel_a(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    affine = np.column_stack((np.ones(len(u)), u))\\n    designs = np.stack([np.column_stack((affine, u[:, j] * u[:, (j + 1) % 4], u[:, j] ** 3)) for j in range(4)])\\n    weights = np.ones(len(u))\\n    for _ in range(25):\\n        root = np.sqrt(weights)\\n        coefs = np.column_stack([np.linalg.lstsq(designs[j] * root[:, None], v[:, j] * root, rcond=None)[0] for j in range(4)])\\n        errors = np.linalg.norm(np.einsum(\'jni,ij->nj\', designs, coefs) - v, axis=1)\\n        scale = max(2 * np.median(errors), 1e-4)\\n        weights = 1 / (1 + (errors / scale) ** 2)\\n    return coefs\\n\\n\\ndef _inverse_a(x, coef):\\n    eye = np.eye(4)\\n    adjacent = np.roll(eye, 1, axis=1)\\n    z = np.linalg.solve(coef[1:5].T, (x - coef[0]).T).T\\n    for _ in range(20):\\n        nxt = np.roll(z, -1, axis=1)\\n        res = coef[0] + z @ coef[1:5] + coef[5] * z * nxt + coef[6] * z ** 3 - x\\n        if np.max(np.abs(res)) < 1e-11:\\n            break\\n        jac = coef[1:5].T + (coef[5] * nxt + 3 * coef[6] * z ** 2)[:, :, None] * eye + (coef[5] * z)[:, :, None] * adjacent\\n        z -= np.linalg.solve(jac, res[..., None])[..., 0]\\n    return z\\n\\n\\ndef _fit_panel_b(panel):\\n    u = panel[\'u\']\\n    return np.linalg.lstsq(np.column_stack((np.ones(len(u)), u)), panel[\'v\'], rcond=None)[0]\\n\\n\\nimport numpy as np\\nfrom scipy.optimize import minimize\\n\\n\\ndef _handoff(q, v, y, traj, configuration):\\n    i, j = np.triu_indices(4)\\n    Q = q[:, i]*q[:, j]\\n    A = np.column_stack((np.ones(len(q)), v))\\n    C = np.linalg.lstsq(np.column_stack((A,Q)),traj,rcond=None)[0]\\n    carrier = np.linalg.svd(C[5:],full_matrices=False)[0][:,0]\\n    carrier /= carrier[1]\\n    g = Q @ carrier\\n    C = np.linalg.lstsq(np.column_stack((A,g)),traj,rcond=None)[0]\\n    L = C[1:5]\\n    mode, w_mix, w_carrier, w_traj = configuration\\n    off = ~np.eye(4,dtype=bool)\\n    keep = np.arange(10)!=1\\n    extra = mode == 2\\n    dm = 12 if mode else 0\\n    dc = 9 if mode else 0\\n    de = 40 if extra else 0\\n    size = 11+dm+dc+de\\n    penal = np.r_[np.full(4,.012),np.full(7,.006),np.full(dm,w_mix),np.full(dc,w_carrier),np.full(de,.24)]**2\\n    y = np.asarray(y).reshape(-1)\\n    F = np.ones((len(q),7))\\n    D = np.ones((len(q),8))\\n\\n    def unpack(p):\\n        M = np.eye(4)\\n        c = carrier.copy()\\n        if mode:\\n            M[off] = p[11:23]\\n            c[keep] += p[23:32]\\n        E = np.outer(c,p[:4])\\n        if extra:\\n            E += p[32:].reshape(10,4)\\n        return M,c,E\\n\\n    def objective(p):\\n        M,c,E = unpack(p)\\n        z = v@M+Q@E\\n        F[:,1:5] = z\\n        F[:,5] = z[:,0]*z[:,2]\\n        F[:,6] = z[:,1]*z[:,3]\\n        theta = p[4:11]\\n        h = np.tanh(F@theta)\\n        D[:,:7] = F\\n        D[:,7] = h\\n        beta = np.linalg.lstsq(D,y,rcond=1e-10)[0]\\n        ry = D@beta-y\\n        T = F[:,:5]\\n        gamma = np.linalg.lstsq(T,traj,rcond=1e-10)[0]\\n        rt = T@gamma-traj\\n        rh = ry*beta[-1]*(1-h*h)\\n        dF = ry[:,None]*beta[:7]+rh[:,None]*theta\\n        dz = dF[:,1:5].copy()+w_traj*(rt@gamma[1:].T)\\n        dz[:,0] += dF[:,5]*z[:,2]\\n        dz[:,2] += dF[:,5]*z[:,0]\\n        dz[:,1] += dF[:,6]*z[:,3]\\n        dz[:,3] += dF[:,6]*z[:,1]\\n        dE = Q.T@dz\\n        grad = penal*p\\n        grad[:4] += c@dE\\n        grad[4:11] += F.T@rh\\n        if mode:\\n            grad[11:23] += (v.T@dz)[off]\\n            grad[23:32] += (dE@p[:4])[keep]\\n        if extra:\\n            grad[32:] += dE.ravel()\\n        return .5*(ry@ry+w_traj*np.sum(rt*rt)+np.dot(penal*p,p)),grad\\n\\n    rng = np.random.default_rng(130)\\n    direction = np.array([0.,1.,1.,1.,-1.,0.,0.])\\n    ds = [np.linalg.solve(L@L.T+r*r*np.eye(4),L@C[5]) for r in (.045,.12,.3)]\\n    ds += [np.eye(4)[k]*(L[k]@C[5])/(L[k]@L[k]) for k in range(4)]\\n    starts = []\\n    for k in range(18):\\n        d = ds[k%len(ds)] if k < 14 else ds[1]+rng.normal(0,.55,4)\\n        theta = direction if k < 7 else rng.normal(0,.85,7)\\n        p = np.r_[d,theta,np.zeros(size-11)]\\n        if mode and k >= 14:\\n            p[11:23] = rng.normal(0,.22,12)\\n        starts.append(p)\\n    best = None\\n    for p in starts:\\n        sol = minimize(objective,p,method=\'L-BFGS-B\',jac=True,options={\'maxiter\':420,\'ftol\':2e-10,\'gtol\':1e-7,\'maxls\':30,\'maxcor\':16})\\n        if best is None or sol.fun < best.fun:\\n            best = sol\\n    M,_,E = unpack(best.x)\\n    return M,E\\n\\n\\n_CONFIGURATION = (2, 0.14, 0.85, 2.0)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train,test = study[\'train\'],study[\'test\']\\n    a,b = _fit_panel_a(study[\'panel_a\']),_fit_panel_b(study[\'panel_b\'])\\n    q,qt = _inverse_a(train[\'x\'],a),_inverse_a(test[\'x\'],a)\\n    v,vt = q@b[1:]+b[0],qt@b[1:]+b[0]\\n    M,E = _handoff(q,v,train[\'y\'],train[\'traj\'],_CONFIGURATION)\\n    i,j = np.triu_indices(4)\\n    return {\'z\':v@M+(q[:,i]*q[:,j])@E,\'z_test\':vt@M+(qt[:,i]*qt[:,j])@E,\'y\':train[\'y\'],\'traj\':train[\'traj\']}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(z):\\n    return np.column_stack((np.ones(len(z)), z))\\n\\n\\ndef _scalar_basis(z, affine):\\n    return np.column_stack((affine, z[:, 0] * z[:, 2], z[:, 1] * z[:, 3]))\\n\\n\\ndef _scalar_design(basis, direction):\\n    return np.column_stack((basis, np.tanh(basis @ direction)))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    data = inputs[\'node_00\']\\n    z, z_test = data[\'z\'], data[\'z_test\']\\n    y = np.asarray(data[\'y\']).reshape(-1)\\n    train, test = _affine(z), _affine(z_test)\\n    basis = _scalar_basis(z, train)\\n\\n    def fit_scalar(direction):\\n        design = _scalar_design(basis, direction)\\n        return design, np.linalg.lstsq(design, y, rcond=None)[0]\\n\\n    def residual(direction):\\n        design, weights = fit_scalar(direction)\\n        return design @ weights - y\\n\\n    start = [0., 1., 1., 1., -1., 0., 0.]\\n    direction = least_squares(residual, start, max_nfev=150).x\\n    _, weights = fit_scalar(direction)\\n    y_hat = _scalar_design(_scalar_basis(z_test, test), direction) @ weights\\n    state_pred = test @ np.linalg.lstsq(train, data[\'traj\'], rcond=None)[0]\\n    last = state_pred[:, -1]\\n    diagnostics = np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred, \'diagnostics_vec\': diagnostics}\\n"}'

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
