"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_07", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_03": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"], "node_03": ["__input__", "node_00", "node_01"]}, "max_fanout": 2, "node_code_shas": {"node_00": "133e3d220d4240c40876fc9b4735fd01eeed02109ccbe9fff46941fe892cfb84", "node_01": "3a429e7f39c801b3746be5ef66fc874a86e98300b9e067ee21b75ff6f53eb499", "node_03": "6e0b58cb2e3475276753b6bbbfbb2e56526d2410a85db8a98bfb34ccbbc1f602"}, "shared_fanout_nodes": ["node_00"], "sink_node_id": "node_03", "topo_order": ["node_00", "node_01", "node_03"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _basis(x):\\n    \\"\\"\\"For each channel, use its own input and its cyclic successor.\\"\\"\\"\\n    z = np.roll(x, -1, axis=1)\\n    return np.stack((np.ones_like(x), x, z, x*z, x**3), axis=-1)\\n\\n\\ndef _calibrate(u, v):\\n    a = _basis(u)\\n\\n    def fit(rows, w):\\n        return np.stack([\\n            np.linalg.lstsq(a[rows, j]*w[:, None], v[rows, j]*w, rcond=None)[0]\\n            for j in range(u.shape[1])\\n        ])\\n\\n    rows = np.arange(len(u))\\n    w = np.ones(len(u))\\n    # Shared Cauchy weights screen contamination at the row level.\\n    for _ in range(20):\\n        c = fit(rows, w)\\n        r = np.linalg.norm(np.sum(a*c, axis=-1)-v, axis=1)\\n        scale = max(2*np.median(r), 1e-9)\\n        w = scale/np.hypot(scale, r)\\n    keep = np.argsort(r)[:-3]\\n    return fit(keep, np.ones(len(keep)))\\n\\n\\ndef _forward(x, c):\\n    return np.sum(_basis(x)*c, axis=-1)\\n\\n\\ndef _jacobian(x, c):\\n    j = np.arange(x.shape[1])\\n    k = (j+1) % x.shape[1]\\n    jac = np.zeros((len(x), len(j), len(j)))\\n    jac[:, j, j] = c[:, 1] + c[:, 3]*x[:, k] + 3*c[:, 4]*x**2\\n    jac[:, j, k] = c[:, 2] + c[:, 3]*x\\n    return jac\\n\\n\\ndef _invert(y, c):\\n    linear = _jacobian(np.zeros_like(y[:1]), c)[0]\\n    x = np.linalg.solve(linear, (y-c[:, 0]).T).T\\n    for _ in range(30):\\n        r = _forward(x, c)-y\\n        if np.max(np.abs(r)) < 1e-11:\\n            break\\n        step = np.linalg.solve(_jacobian(x, c), r[:, :, None])[:, :, 0]\\n        trial = x-step\\n        norm = np.linalg.norm(r, axis=1)\\n        for _ in range(12):\\n            bad = np.linalg.norm(_forward(trial, c)-y, axis=1) > norm\\n            if not np.any(bad):\\n                break\\n            step[bad] *= 0.5\\n            trial[bad] = x[bad]-step[bad]\\n        x = trial\\n    return x\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    a, c = study[\'panel_a\'], study[\'panel_c\']\\n    coef = _calibrate(a[\'u\'], a[\'v\'])\\n    measured = np.concatenate((study[\'train\'][\'x\'], study[\'test\'][\'x\']))\\n    return {\'u\': c[\'u\'], \'v1\': c[\'v1\'], \'v2\': c[\'v2\'], \'x\': _invert(measured, coef)}\\n", "node_01": "import numpy as np\\n\\n\\ndef _basis(x):\\n    return np.column_stack((np.ones(len(x)), *(x[:, i]*x[:, j] for i in range(x.shape[1]) for j in range(i + 1, x.shape[1]))))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    panel = inputs[\'node_00\']\\n    coef = np.linalg.lstsq(_basis(panel[\'u\']), panel[\'v1\'], rcond=None)[0]\\n    return {\'b\': _basis(panel[\'x\']) @ coef}\\n", "node_03": "import numpy as np\\nfrom scipy.linalg import cho_factor, cho_solve\\nfrom scipy.optimize import minimize\\nfrom itertools import combinations, combinations_with_replacement\\n\\n\\ndef _cosine_basis(x):\\n    return np.column_stack((np.ones(len(x)), np.cos(2.4*x)))\\n\\n\\ndef _features(study, panel, b):\\n    x = panel[\'x\']\\n    u = panel[\'u\']\\n    c = _cosine_basis(x) @ np.linalg.lstsq(_cosine_basis(u), panel[\'v2\'], rcond=None)[0]\\n    pb = study[\'panel_b\']\\n    d = np.c_[np.ones(len(x)), x] @ np.linalg.lstsq(np.c_[np.ones(len(pb[\'u\'])), pb[\'u\']], pb[\'v\'], rcond=None)[0]\\n    f = np.cos(2.4*x)\\n    h = np.sin(2.4*x)\\n    def cross(a, b):\\n        return (a[:, :, None]*b[:, None, :]).reshape(len(x), -1)\\n    def poly(a, k):\\n        return np.column_stack([np.prod(a[:, ix], axis=1) for ix in combinations_with_replacement(range(a.shape[1]), k)])\\n    def ratios(a, b):\\n        return (a[:, :, None]/(1+b[:, None, :]**2)).reshape(len(x), -1)\\n    groups = [np.ones((len(x),1)), b, c, x, x*x, poly(x,2), poly(x,3), poly(x,4), b*c, cross(b,c), poly(b,2), poly(c,2), cross(x,b), cross(x,c), cross(h,b), cross(h,c), h, f*f, cross(f,b), poly(f,2), cross(x*x,b), cross(x*x,c), cross(b,b*c), cross(c,b*c), cross(b,np.tanh(c)), cross(c,np.tanh(b)), ratios(b,c), ratios(c,b), np.tanh(b+c), np.tanh(b-c), np.tanh(b), np.tanh(c), d*c, d*b, cross(d,c), cross(d,b), np.cos(2.4*(x+np.roll(x,1,axis=1))), np.sin(2.4*(x+np.roll(x,1,axis=1))), np.cos(2.4*(d-d.mean(axis=0))), np.tanh(d), f*x, cross(f,h)]\\n    return x, groups\\n\\n\\ndef _fit(study, panel, b, mode):\\n    x, groups = _features(study, panel, b)\\n    obs = study[\'train\']\\n    y = np.column_stack((obs[\'y\'], obs[\'traj\']))\\n    n = len(y)\\n    means = y.mean(axis=0)\\n    y = y-means\\n    if mode in (\'base\', \'ridge\'):\\n        if mode == \'base\':\\n            a = np.column_stack(groups[:3]); return means+a[n:]@np.linalg.lstsq(a[:n],y,rcond=None)[0]\\n        groups = groups[:24]\\n    kernels = []\\n    columns = []\\n    for g in groups[1:]:\\n        sd = g[:n].std(axis=0)\\n        keep = sd>1e-8\\n        if not keep.any(): continue\\n        z = (g[:,keep]-g[:n,keep].mean(axis=0))/sd[keep]\\n        columns.append(z)\\n        kernels.append((z@z[:n].T)/z.shape[1])\\n    dx=x[:,None,:]-x[None,:n,:]; distance=np.sum(dx*dx,axis=-1)\\n    for ell in (.4,.8,1.5,3.):\\n        kernels.append(np.exp(-distance/(2*ell**2)))\\n    kernels=np.array(kernels)\\n    if mode == \'ridge\':\\n        K = kernels.mean(axis=0)\\n        return means+K[n:]@np.linalg.solve(K[:n]+.004*np.eye(n),y)\\n    _, sv, vectors=np.linalg.svd(y,full_matrices=False)\\n    k=3 if mode==\'eb3\' else 4\\n    scale = sv[:k]**(.65 if mode==\'eb4\' else .9)\\n    z=(y@vectors[:k].T)/scale\\n    z=z/np.sqrt(np.mean(z*z))\\n    stack = np.concatenate((kernels[:,:n,:],np.eye(n)[None]),axis=0)\\n    flat = stack.reshape(len(stack),-1)\\n    rand = np.random.default_rng(5)\\n    def loss(w):\\n        p=np.exp(w); K=(p@flat).reshape(n,n)\\n        fac=cho_factor(K,check_finite=False); a=cho_solve(fac,z,check_finite=False)\\n        v=.5*np.sum(z*a)+k*np.log(np.diag(fac[0])).sum()\\n        M=k*cho_solve(fac,np.eye(n),check_finite=False)-a@a.T\\n        grad=.5*p*(flat@M.ravel())\\n        return v,grad\\n    best=None\\n    for i in range(2):\\n        init=np.r_[np.full(len(kernels),-4.),-5.]\\n        if i: init+=rand.normal(0,.8,len(init))\\n        res=minimize(loss,init,jac=True,method=\'L-BFGS-B\',bounds=[(-13.,4.)]*len(init),options={\'maxiter\':125})\\n        if best is None or res.fun<best.fun: best=res\\n    p=np.exp(best.x); K=np.tensordot(p[:-1],kernels,axes=1)\\n    pred=means+K[n:]@np.linalg.solve(K[:n]+p[-1]*np.eye(n),y)\\n    return pred\\n\\n\\n\\ndef _adaptive(study, panel, b, variant):\\n    rank, power, family, refinement, target = variant\\n    x, groups = _features(study, panel, b)\\n    n = len(study[\'train\'][\'y\'])\\n    raw = np.column_stack((study[\'train\'][\'y\'], study[\'train\'][\'traj\']))\\n    mean = raw.mean(axis=0)\\n    Y = raw-mean\\n    if target == 1:\\n        source = Y[:,1:]\\n    elif target == 2:\\n        source = np.c_[Y[:,0]/Y[:,0].std(), Y[:,1:]/np.sqrt(np.mean(Y[:,1:]**2))]\\n    else:\\n        source = Y\\n    U, sv, V = np.linalg.svd(source, full_matrices=False)\\n    r = min(rank,len(sv))\\n    Z = U[:,:r]*sv[:r]**(1-power)\\n    Z /= np.sqrt(np.mean(Z**2))\\n    if family:\\n        c = groups[2]\\n        f = np.cos(2.4*x)\\n        h = np.sin(2.4*x)\\n        def cross(a, q):\\n            return (a[:,:,None]*q[:,None,:]).reshape(len(a),-1)\\n        def powers(a, k):\\n            return np.column_stack([np.prod(a[:,ii],axis=1) for ii in combinations_with_replacement(range(a.shape[1]),k)])\\n        q = np.column_stack((b,c))\\n        p = np.column_stack([x[:,i]*x[:,j] for i,j in combinations(range(x.shape[1]),2)])\\n        groups += [cross(p,f),cross(p,c),cross(p,b),cross(f,c),cross(f,h),cross(f,x),cross(h,x),powers(q,3),cross(x,powers(q,2)),cross(x**3,f),cross(powers(x,2),f),cross(powers(x,2),h),cross(x,f*f),cross(p,f*f),cross(powers(f,2),b),cross(powers(f,2),c),cross(powers(f,2),x)]\\n        for g in (.5,1.,2.):\\n            groups += [np.tanh(g*q), np.sin(g*q), np.cos(g*q),cross(np.tanh(g*b),c),cross(np.tanh(g*c),b),cross(np.sin(g*b),c),cross(np.sin(g*c),b)]\\n        if family==2:\\n            groups += [cross(np.cos(j*x),p) for j in (1.2,1.8,3.,3.6,4.8)]\\n            groups += [cross(np.sin(j*x),p) for j in (1.2,1.8,3.,3.6,4.8)]\\n            groups += [np.column_stack([fun(g*(q[:,i]+d*q[:,j])) for i,j in combinations(range(q.shape[1]),2)]) for fun in (np.sin,np.cos,np.tanh) for d in (-1,1) for g in (1.,2.)]\\n    cols=[]; kernels=[]\\n    for g in groups[1:]:\\n        sd=g[:n].std(axis=0); keep=sd>1e-8\\n        if not keep.any(): continue\\n        a=(g[:,keep]-g[:n,keep].mean(axis=0))/sd[keep]; a/=np.sqrt(a.shape[1]);cols.append(a)\\n        kernels.append(a@a[:n].T)\\n    dist=np.sum((x[:,None,:]-x[None,:n,:])**2,axis=2)\\n    for ell in (.4,.8,1.5,3.): kernels.append(np.exp(-dist/(2*ell**2)))\\n    def learn(K,z,start=None):\\n        stack=np.concatenate((K[:,:n,:],np.eye(n)[None]),axis=0).reshape(len(K)+1,-1)\\n        nz=z.shape[1]; eye=np.eye(n)\\n        def fun(w):\\n            p=np.exp(w); cov=np.einsum(\'i,ij->j\',p,stack).reshape(n,n)\\n            fac=cho_factor(cov,check_finite=False); a=cho_solve(fac,z,check_finite=False)\\n            G=nz*cho_solve(fac,eye,check_finite=False)-a@a.T\\n            return .5*np.sum(z*a)+nz*np.log(np.diag(fac[0])).sum(), .5*p*np.einsum(\'ij,j->i\',stack,G.ravel())\\n        if start is None: start=np.r_[np.full(len(K),-4.),-5.]\\n        opt=minimize(fun,start,jac=True,method=\'L-BFGS-B\',bounds=[(-15.,5.)]*len(start),options={\'maxiter\':165,\'ftol\':2e-9})\\n        return np.exp(opt.x)\\n    K=np.array(kernels); weights=learn(K,Z)\\n    if refinement:\\n        order=np.argsort(weights[:len(cols)])[::-1]; additions=[]\\n        for j in order[:refinement]:\\n            a=cols[j]\\n            additions.extend([np.outer(v,v[:n])*a.shape[1] for v in a.T])\\n        K=np.concatenate((K,np.array(additions)),axis=0)\\n        start=np.log(np.r_[weights[:-1],np.full(len(additions),1e-4),weights[-1]])\\n        weights=learn(K,Z,start)\\n    cov=np.einsum(\'i,ijk->jk\',weights[:-1],K)\\n    return mean+cov[n:]@np.linalg.solve(cov[:n]+weights[-1]*np.eye(n),Y)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    s,p,b=inputs[\'__input__\'],inputs[\'node_00\'],inputs[\'node_01\'][\'b\']\\n    a=_fit(s,p,b,\'eb3\') if _TRAJ_CONFIG is None or _SCALAR_CONFIG is None else None\\n    y=a[:,0] if _SCALAR_CONFIG is None else _adaptive(s,p,b,_SCALAR_CONFIG)[:,0]\\n    if _TRAJ_CONFIG == _SCALAR_CONFIG and _TRAJ_CONFIG is not None:\\n        pred=_adaptive(s,p,b,_TRAJ_CONFIG)\\n        y,state=pred[:,0],pred[:,1:]\\n    elif _TRAJ_CONFIG is None:\\n        state=a[:,1:]\\n    elif isinstance(_TRAJ_CONFIG,list):\\n        state=np.mean([_adaptive(s,p,b,q)[:,1:] if q is not None else _fit(s,p,b,\'eb3\')[:,1:] for q in _TRAJ_CONFIG],axis=0)\\n    else:\\n        state=_adaptive(s,p,b,_TRAJ_CONFIG)[:,1:]\\n    return {\'y_hat\':y,\'state_pred\':state,\'diagnostics_vec\':np.array([y.mean(),y.std(),state[:,-1].mean(),state[:,-1].std()])}\\n\\n\\n_TRAJ_CONFIG = (2, 0.6, 1, 0, 1)\\n_SCALAR_CONFIG = None\\n"}'

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
