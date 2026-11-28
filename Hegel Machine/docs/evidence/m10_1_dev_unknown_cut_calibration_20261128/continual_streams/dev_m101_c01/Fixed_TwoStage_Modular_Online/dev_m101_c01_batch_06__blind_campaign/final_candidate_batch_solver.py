"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_04", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "7abe63231c58b13076d2407f829505ae29fba10ee45430b78c65114c68fc0def"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Calibrate the inputs and fit separate scalar and time-dependent readouts.\\"\\"\\"\\nimport numpy as np\\nfrom itertools import combinations_with_replacement as cw\\nfrom scipy.linalg import cho_factor, cho_solve\\nfrom scipy.optimize import minimize, minimize_scalar\\n\\ndef poly(x,k):return np.column_stack([np.prod(x[:,j],axis=1) for j in cw(range(x.shape[1]),k)])\\ndef design(x,k):return np.column_stack([poly(x,i) for i in range(k+1)])\\ndef robust(X,y):\\n rng=np.random.default_rng(123); m=X.shape[1]; best=None\\n for _ in range(900):\\n  j=rng.choice(len(X),m,replace=False); w=np.linalg.lstsq(X[j],y[j],rcond=None)[0]; r=np.linalg.norm(y-X@w,axis=1); i=np.argsort(r); score=r[i[int(.75*len(X))]]\\n  if best is None or score<best[0]:best=score,i\\n j=best[1][:int(.8*len(X))]; w=np.linalg.lstsq(X[j],y[j],rcond=None)[0]\\n for _ in range(3):\\n  r=np.linalg.norm(y-X@w,axis=1); j=r<max(4*np.median(r),.0005); w=np.linalg.lstsq(X[j],y[j],rcond=None)[0]\\n return w,np.linalg.norm(y-X@w,axis=1)\\ndef predict(x,q,y,groups):\\n n=len(x); xs=np.r_[x,q]; D=design(xs,2); D[:,5:]*=.5\\n K=[D[:,:1]@D[:,:1].T,D[:,1:5]@D[:,1:5].T,D[:,5:]@D[:,5:].T]\\n for a in groups:\\n  a=np.asarray(a); a=(a-a[:n].mean(0))/np.maximum(.02,a[:n].std(0)); K.append(a@a.T/a.shape[1]);\\n K=np.asarray(K); train=K[:,:n,:n]; cross=K[:,n:,:n]; eye=np.eye(n); Y=np.asarray(y); center=Y.mean(0); scale=np.maximum(1e-8,Y.std(0)); Y=(Y-center)/scale\\n if Y.ndim==1:Y=Y[:,None]\\n def obj(p,ret=False):\\n  w=np.exp(p); M=np.einsum(\'i,ijk->jk\',w[:-1],train)+w[-1]*eye; fac=cho_factor(M,check_finite=False); c=cho_solve(fac,Y,check_finite=False)\\n  if ret:return (np.einsum(\'i,ijk->jk\',w[:-1],cross)@c).squeeze()*scale+center\\n  G=Y.shape[1]*cho_solve(fac,eye,check_finite=False)-c@c.T\\n  v=.5*np.sum(Y*c)+Y.shape[1]*np.log(fac[0].diagonal()).sum(); g=np.r_[np.einsum(\'ijk,jk->i\',train,G),np.trace(G)]*w*.5\\n  return v,g\\n p=minimize(obj,np.r_[np.full(3,-1.),np.full(len(K)-3,-5.),-8.],jac=True,method=\'L-BFGS-B\',bounds=[(-13,2)]*len(K)+[(-12,-2)],options={\'maxiter\':100}).x\\n return obj(p,True)\\ndef group(q,kind):\\n out=[]\\n for u in q:\\n  if kind==\'smooth\':\\n   out += [np.c_[np.sin(1.6*u),np.cos(1.6*u)],np.c_[np.sin(2.8*u),np.cos(2.8*u)],poly(u,3),poly(u,4)]\\n  if kind==\'soft\':\\n   out += [u/(1+.83*abs(u)),(u[:,:,None]*(u/(1+.83*abs(u)))[:,None,:]).reshape(len(u),-1),poly(u/(1+.83*abs(u)),2)]\\n  if kind==\'mixed\':\\n   v=np.c_[u/(1+.83*abs(u)),np.cos(1.29*u),np.sin(1.6*u)];out += [v,(u[:,:,None]*v[:,None,:]).reshape(len(u),-1),poly(u,3),poly(u,4)]\\n return out\\ndef amplitude(t,Z):\\n def fit(r):\\n  b=-np.expm1(-r*t); c=Z@b/(b@b);return c,b\\n def obj(r):\\n  c,b=fit(r);return np.sum((Z-c[:,None]*b)**2)\\n r=minimize_scalar(obj,bounds=(.3,1.5),method=\'bounded\').x\\n return r,*fit(r)\\n\\ndef spaces(s,x):\\n a=s[\'panel_a\'];A,_=robust(design(a[\'u\'],1),a[\'v\']);A[1:]=np.diag(A[1:].diagonal()); b=s[\'panel_b\'];B=np.linalg.lstsq(design(b[\'u\'],1),b[\'v\'],rcond=None)[0]\\n af=lambda X,M:X@M[1:]+M[0];ai=lambda X,M:(X-M[0])@np.linalg.inv(M[1:]);\\n return {\'x\':x,\'a\':af(x,A),\'ai\':ai(x,A),\'b\':af(x,B),\'bi\':ai(x,B),\'ba\':af(af(x,A),B),\'bai\':af(ai(x,A),B),\'bia\':ai(af(x,A),B),\'biai\':ai(ai(x,A),B),\'ab\':af(af(x,B),A),\'aib\':ai(af(x,B),A),\'abi\':af(ai(x,B),A),\'aibi\':ai(ai(x,B),A)}\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    train = study[\'train\']\\n    x, y, z = (train[k] for k in (\'x\', \'y\', \'traj\'))\\n    q = study[\'test\'][\'x\']\\n    t = study[\'eval_times\']\\n    sp = spaces(study, np.r_[x, q])\\n    g1 = group([sp[\'b\']], \'smooth\')\\n    g2 = group([sp[\'ba\']], \'mixed\')\\n    g3 = group([sp[\'ab\']], \'smooth\')\\n    g4 = group([sp[\'b\']], \'mixed\')\\n    yh = (predict(x,q,y,g1) + predict(x,q,y,g2)) / 2\\n    _, a, b = amplitude(t, z)\\n    pa = .55*predict(x,q,a,g4) + .45*predict(x,q,a,g3)\\n    residual = z - a[:, None]*b\\n    v = np.linalg.svd(residual, full_matrices=False)[2][0]\\n    pc = predict(x,q,residual@v,g4)\\n    state = pa[:, None]*b + pc[:, None]*v\\n    end = state[:, -1]\\n    return {\'y_hat\': yh, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([yh.mean(), yh.std(), end.mean(), end.std()])}\\n"}'

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
