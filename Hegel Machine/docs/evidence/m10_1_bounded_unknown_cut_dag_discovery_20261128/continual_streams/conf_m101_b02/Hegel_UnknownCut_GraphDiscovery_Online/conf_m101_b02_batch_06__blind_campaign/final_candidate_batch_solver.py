"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_08", "depth": 3, "fanout_by_node": {"node_00": 1, "node_01": 0, "node_02": 1}, "incoming": {"node_00": ["node_02"], "node_01": ["__input__", "node_00"], "node_02": ["__input__"]}, "max_fanout": 1, "node_code_shas": {"node_00": "443bb09a004711496a5996e394924bd593fe2aa46aaf91c073fba5cd3a6d6be4", "node_01": "3ed491ae082f690b317caba0ec7bf9308c0136d6e69d3a0fc8a1c0cf8a548708", "node_02": "90fc8e5c8b500f4700024cfa350825dc9881d321313f12ef132a006d9f68cb8f"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_02", "node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _basis(u, q):\\n    return np.column_stack((np.ones(len(u)), u, np.tanh(q[0]*u), u/(1+q[1]*u*u)))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    p = inputs[\'node_02\']\\n    u, v, z = p[\'b_u\'], p[\'b_v\'], p[\'z\']\\n\\n    def fit(q):\\n        a = _basis(u, q)\\n        return a, np.linalg.lstsq(a, v, rcond=None)[0]\\n\\n    def residual(q):\\n        a, w = fit(q)\\n        return (a @ w - v).ravel()\\n\\n    q = least_squares(residual, [3.5, 6.], bounds=([.1, .1], [8., 15.])).x\\n    _, w = fit(q)\\n    return {\'z\': z, \'b\': _basis(z, q) @ w}\\n", "node_01": "import numpy as np\\nfrom scipy.linalg import cho_factor, cho_solve\\nfrom scipy.optimize import minimize\\n\\n\\ndef _one(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _cycle(x):\\n    return x * np.roll(x, -1, axis=1)\\n\\n\\ndef _panels(u, v1, v2, x):\\n    def calibrate(f, v):\\n        return f(x) @ np.linalg.lstsq(f(u), v, rcond=None)[0]\\n    return calibrate(lambda a: _one(_cycle(a)), v1), calibrate(lambda a: _one(np.c_[a, a*a]), v2)\\n\\n\\ndef _kernels(x, z, b, c, d, n):\\n    groups = []\\n    def std(a):\\n        return (a-a[:n].mean(0))/np.maximum(a[:n].std(0), 1e-10)\\n    def add(a):\\n        a = std(a)\\n        groups.append(a @ a.T / a.shape[1])\\n    panels = [x, z, b, c, d]\\n    for f in panels:\\n        for j in range(f.shape[1]):\\n            q = f[:, j:j+1]\\n            a = std(q)\\n            add(a)\\n            add(a*a)\\n            add(np.c_[np.sin(a), np.tanh(1.6*a), a/(1+2*a*a), a**3])\\n            add(np.c_[np.cos(2*a), np.sin(2.5*a), a**4])\\n    for p, f in enumerate(panels):\\n        for r, g in enumerate(panels[:p+1]):\\n            f0, g0 = std(f), std(g)\\n            ix = [(i,j) for i in range(4) for j in range(4) if r!=p or i<j]\\n            for subset in (ix, [(i,j) for i,j in ix if (i-j)%4 in (1,3)]):\\n                add(np.column_stack([f0[:,i]*g0[:,j] for i,j in subset]))\\n            if p < 4 and r < 3:\\n                add(np.column_stack([np.tanh(1.4*f0[:,i])*g0[:,j] for i,j in ix]))\\n    for p, f in enumerate(panels[:4]):\\n        a=std(f); k=a@a.T/4\\n        groups.extend([(1+k)**3/8, (1+k)**4/16])\\n        for g in panels[:p]:\\n            h=std(g); l=h@h.T/4\\n            groups.append((1+k+l)**3/12)\\n    a=std(np.c_[x,z,b,c]); k=a@a.T/a.shape[1]; dist=k.diagonal()[:,None]+k.diagonal()[None,:]-2*k\\n    groups.extend([(1+k)**2/4, (1+k)**3/8, np.exp(-.3*np.maximum(dist,0)), np.exp(-np.maximum(dist,0))])\\n    return np.array(groups)\\n\\n\\ndef _predict(k, y, noise, penalty):\\n    n=len(y); m=len(k); mu=y.mean(0); scale=np.maximum(y.std(0),1e-10); y=(y-mu)/scale\\n    a=np.concatenate((np.eye(n)[None],k[:,:n,:n])); flat=np.ascontiguousarray(a.reshape(m+1,-1)); cross=k[:,n:,:n]; inv0=np.eye(n)\\n    prior=np.log(np.r_[noise,np.full(m,1./m)])\\n    def calc(theta):\\n        v=np.exp(theta); cov=(np.einsum(\'i,ij->j\',v,flat)).reshape(n,n)+1.\\n        cf=cho_factor(cov,check_finite=False)\\n        alpha=cho_solve(cf,y,check_finite=False)\\n        ic=cho_solve(cf,inv0,check_finite=False)\\n        loss=np.log(cf[0].diagonal()).sum()+.5*np.sum(alpha*y)/y.shape[1]\\n        grad=.5*v*(np.einsum(\'ij,j->i\',flat,(ic-alpha@alpha.T/y.shape[1]).ravel()))\\n        delta=theta-prior\\n        return loss+penalty*(delta@delta),grad+2*penalty*delta\\n    theta=minimize(calc,prior,jac=True,method=\'L-BFGS-B\',bounds=[(-12.,1.)]+[(-15.,3.)]*m,options={\'maxiter\':85,\'ftol\':2e-7}).x\\n    v=np.exp(theta); cov=(np.einsum(\'i,ij->j\',v,flat)).reshape(n,n)+1.\\n    alpha=cho_solve(cho_factor(cov,check_finite=False),y,check_finite=False)\\n    return ((1.+np.einsum(\'k,kij->ij\',v[1:],cross))@alpha)*scale+mu\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    s=inputs[\'__input__\']; z=inputs[\'node_00\'][\'z\']; b=inputs[\'node_00\'][\'b\']\\n    train=s[\'train\']; x=np.vstack((train[\'x\'],s[\'test\'][\'x\'])); n=len(train[\'y\'])\\n    panel=s[\'panel_c\']; c,d=_panels(panel[\'u\'],panel[\'v1\'],panel[\'v2\'],b)\\n    k=_kernels(x,z,b,c,d,n)\\n    times=s[\'eval_times\']\\n    h=np.column_stack((-np.expm1(-1.56*times),-np.expm1(-.38*times),np.ones(len(times))))\\n    q=np.linalg.qr(h)[0]\\n    modes=train[\'traj\']@q\\n    # Share covariance relevance between the scalar and dominant time mode.\\n    joint=np.column_stack((train[\'y\'],modes[:,0]))\\n    prediction=0.\\n    for penalty, weight in ((0.004, 0.5), (0.002, 0.5)):\\n        prediction=prediction+weight*np.column_stack((\\n            _predict(k,joint,.004,penalty),\\n            _predict(k,modes[:,1:2],.004,penalty),\\n            _predict(k,modes[:,2:3],.04,penalty)))\\n    y_hat=prediction[:,0]\\n    state_pred=prediction[:,1:]@q.T\\n    final=state_pred[:,-1]\\n    return {\'y_hat\':y_hat,\'state_pred\':state_pred,\'diagnostics_vec\':np.array([y_hat.mean(),y_hat.std(),final.mean(),final.std()])}\\n", "node_02": "import numpy as np\\n\\n\\ndef _design(u):\\n    return np.stack((np.ones_like(u), u, np.roll(u, -1, axis=1),\\n                     u * np.roll(u, -1, axis=1), u**3), axis=-1)\\n\\n\\ndef _calibrate(u, v):\\n    basis = _design(u)\\n    weight = np.ones(len(u))\\n    for _ in range(18):\\n        coeff = np.stack([np.linalg.lstsq(basis[:, j] * weight[:, None],\\n                                         v[:, j] * weight, rcond=None)[0]\\n                          for j in range(u.shape[1])])\\n        residual = np.linalg.norm(np.sum(basis * coeff, axis=-1) - v, axis=1)\\n        scale = max(np.median(residual), 1e-8)\\n        weight = 1 / np.hypot(1, residual / scale)\\n    return coeff\\n\\n\\ndef _invert(x, coeff):\\n    j = np.arange(x.shape[1])\\n    k = (j + 1) % len(j)\\n    linear = np.diag(coeff[:, 1])\\n    linear[j, k] = coeff[:, 2]\\n    z = np.linalg.solve(linear, (x - coeff[:, 0]).T).T\\n    jac = np.zeros((len(x), len(j), len(j)))\\n    for _ in range(24):\\n        r = np.sum(_design(z) * coeff, axis=-1) - x\\n        if np.max(np.abs(r)) < 1e-11:\\n            break\\n        jac[:, j, j] = coeff[:, 1] + coeff[:, 3] * z[:, k] + 3 * coeff[:, 4] * z**2\\n        jac[:, j, k] = coeff[:, 2] + coeff[:, 3] * z\\n        dz = np.linalg.solve(jac, r[..., None])[..., 0]\\n        z -= dz / np.maximum(1, np.max(np.abs(dz), axis=1, keepdims=True))\\n    return z\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    s = inputs[\'__input__\']\\n    a, b = s[\'panel_a\'], s[\'panel_b\']\\n    coeff = _calibrate(a[\'u\'], a[\'v\'])\\n    x = np.vstack((s[\'train\'][\'x\'], s[\'test\'][\'x\']))\\n    return {\'z\': _invert(x, coeff), \'b_u\': b[\'u\'], \'b_v\': b[\'v\']}\\n"}'

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
