"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_01", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "7a0d2bd0d767de4a3cc3a15d9fb881ad75c1c42e2dcff1b1f1bee75f7b2c6f5d"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Predict scalar and trajectory outputs from the measured training pairs.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.linalg import cho_factor, cho_solve\\nfrom scipy.optimize import minimize\\nfrom itertools import combinations_with_replacement\\n\\ndef polys(x,k):\\n return np.column_stack([np.prod(x[:,j],axis=1) for j in combinations_with_replacement(range(4),k)])\\n\\ndef gp(x,y,q,kind):\\n xq=np.r_[x,q]; n=len(x); xq=xq/.7\\n y=np.asarray(y); y=y[:,None] if y.ndim==1 else y\\n y0=y.mean(0); sy=np.maximum(y.std(0),1e-8); ymm=(y-y0)/sy\\n Q=[polys(xq,k) for k in range(1,5)]; W=[p@p.T for p in Q]; D=(xq[:,None,:]-xq[None,:,:])**2; Dtr=D[:n,:n]; eye=np.eye(n)\\n trW=[w[:n,:n] for w in W]; const=np.ones((len(xq),len(xq)))\\n if kind==\'poly\': init=[-.3,-2,-4,-6,-8]; bound=[(-9,2)]*4+[(-13,-3)]\\n else: init=[-.3,-2,-3,0,0,0,0,-8]; bound=[(-9,2)]*3+[(-1.8,2)]*4+[(-13,-3)]\\n def matrices(p,train):\\n  a=np.exp(p); w=trW if train else W; c=const[:n,:n] if train else const[n:,:n]; w=w if train else [b[n:,:n] for b in w]\\n  if kind==\'poly\':\\n   gg=[a[i]*w[i] for i in range(4)]; k=c+sum(gg)\\n  else:\\n   delta=Dtr if train else D[n:,:n]; dt=delta*a[3:7]; dist=dt.sum(-1); g=np.exp(-.5*dist)\\n   if kind==\'rq\': g=(1+.5*dist)**-1\\n   a0,a1,a2=a[:3]; k=c+a0*w[0]+a1*w[1]+a2*g; gg=[a0*w[0],a1*w[1],a2*g]; gg.extend(np.moveaxis(-.5*a2*g[:,:,None]*dt*(g[:,:,None] if kind==\'rq\' else 1),2,0))\\n  if train: k=k+a[-1]*eye; gg.append(a[-1]*eye)\\n  return k,gg\\n def loss(p):\\n  k,g=matrices(p,True); fac=cho_factor(k,lower=True,check_finite=False); c=cho_solve(fac,ymm,check_finite=False); inv=cho_solve(fac,eye,check_finite=False); J=y.shape[1]*inv-c@c.T\\n  prior=np.zeros(len(p)); prior[3:7]=.10 if kind!=\'poly\' else 0\\n  return .5*np.sum(c*ymm)+y.shape[1]*np.log(np.diag(fac[0])).sum()+np.sum(prior*p*p), .5*np.einsum(\'ij,kij->k\',J,g)+2*prior*p\\n fit=minimize(loss,init,jac=True,method=\'L-BFGS-B\',bounds=bound,options={\'maxiter\':160,\'ftol\':1e-10}); k,_=matrices(fit.x,True); cross,_=matrices(fit.x,False)\\n return y0+sy*(cross@cho_solve(cho_factor(k,lower=True),ymm)),fit.x\\n\\nfrom scipy.optimize import least_squares\\nfrom itertools import combinations_with_replacement as comb\\n\\ndef poly(x,d,start=0):\\n return np.column_stack([np.prod(x[:,j],axis=1) for k in range(start,d+1) for j in comb(range(4),k)])\\n\\ndef rat(x,y,q,kind):\\n order,mode,reg=kind; X,T=poly(x,order),poly(q,order); Z,W=poly(x,2,1),poly(q,2,1)\\n n=len(x); r2=np.sum(x*x,1); r2q=np.sum(q*q,1)\\n def denominator(p,V,r):\\n  if mode==\'rad\': return 1+(p[0]+V[:,:4]@p[1:5]+r*p[5])**2\\n  if mode==\'diag\': return 1+(p[0]+V[:,:4]@p[1:5]+V[:,[4,8,11,13]]@p[5:])**2\\n  if mode==\'full\': return 1+(p[0]+V@p[1:])**2\\n  if mode==\'rad2\': return 1+(p[0]+V[:,:4]@p[1:5])**2+np.exp(p[5])*r\\n  if mode==\'rad3\': return 1+(p[0]+V[:,:4]@p[1:5])**2+np.exp(p[5])*r*r\\n  if mode==\'qrad\': return np.exp(np.r_[p[:4],p[4]]@np.c_[V[:,:4],r].T)\\n  return 1+np.sum((V[:,:4]@p[:8].reshape(4,2)+p[8:])**2,axis=1)\\n m={\'rad\':6,\'diag\':9,\'full\':15,\'rad2\':6,\'rad3\':6,\'qrad\':5,\'rank2\':10}[mode]\\n regX=np.r_[np.full(15,.0001),np.full(X.shape[1]-15,reg)]; regZ=np.r_[np.full(min(5,m),.0005),np.full(max(0,m-5),reg)]\\n def residual(p,ret=False):\\n  P=X/denominator(p,Z,r2)[:,None]; c=np.linalg.lstsq(np.r_[P,np.diag(regX)],np.r_[y,np.zeros(len(regX))],rcond=None)[0]; return c if ret else np.r_[P@c-y,regX*c,regZ*p]\\n rng=np.random.default_rng(48); starts=[rng.normal(0,.25,m) for _ in range(5)]; starts.append(np.r_[.45,0,0,.25,.15,.10,np.zeros(max(0,m-6))][:m]);\\n if mode in (\'rad2\',\'rad3\'):\\n  for p in starts:p[5]=-2\\n best=min((least_squares(residual,p,bounds=(-8,8),max_nfev=100,ftol=2e-7) for p in starts),key=lambda f:f.cost)\\n return (T@residual(best.x,True))/denominator(best.x,W,r2q), np.mean(abs(best.fun[:n])),best.x\\n\\n\\ndef _components(x, y, z, q):\\n    scalar = np.column_stack([\\n        rat(x, y, q, (2, \'rad3\', .035))[0],\\n        rat(x, y, q, (3, \'rad2\', .25))[0],\\n        rat(x, y, q, (3, \'qrad\', .25))[0],\\n        gp(x, y, q, \'rq\')[0][:, 0],\\n    ])\\n    basis = np.linalg.svd(z, full_matrices=False)[2][:2]\\n    amplitudes = z @ basis.T\\n    state = np.stack([\\n        gp(x, amplitudes, q, \'poly\')[0] @ basis,\\n        gp(x, amplitudes, q, \'rbf\')[0] @ basis,\\n        np.column_stack([gp(x, a, q, \'rbf\')[0][:, 0] for a in amplitudes.T]) @ basis,\\n    ])\\n    return scalar, state\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    train = study[\'train\']\\n    scalar, state = _components(train[\'x\'], train[\'y\'], train[\'traj\'], study[\'test\'][\'x\'])\\n    y_hat = scalar @ np.array([.25, .30, .30, .15])\\n    state_pred = np.average(state, axis=0, weights=[.4, .2, .4])\\n    end = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])}\\n"}'

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
