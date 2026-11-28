"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_13", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "fb8e6333b3879b45ba25ae15aaca6fbca1a86cd1cb8aad3df38910d0cb5cfc8b"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Calibrated composed operators with separate scalar and trajectory readouts.\\"\\"\\"\\nimport numpy as np\\n\\ndef D(x): return np.c_[np.ones(len(x)), x]\\ndef fit(x,y): return np.linalg.lstsq(x,y,rcond=1e-10)[0]\\ndef P(x): return x*np.roll(x,-1,axis=1)\\ndef AF(x): return D(np.c_[x,P(x),x**3])\\ndef B(x): return D(np.sign(x)*np.maximum(abs(x)-.35,0)**1.5)\\ndef E(x): return D(np.sqrt(1+.924227*x*x)+.831839*x)\\n\\ndef features(s,x):\\n    a=s[\'panel_a\']; u,v=a[\'u\'],a[\'v\']; F=AF(u); k=np.ones(len(u),bool)\\n    for _ in range(8):\\n        w=fit(F[k],v[k]); r=np.linalg.norm(F@w-v,axis=1); h=r<max(.002,3*np.median(r))\\n        if h.sum()<F.shape[1] or np.array_equal(h,k): break\\n        k=h\\n    z=np.linalg.solve(w[1:5].T,(x-w[0]).T).T\\n    for _ in range(14):\\n        J=w[1:5]+np.roll(z,-1,axis=1)[:,:,None]*w[5:9]+np.roll(z[:,:,None]*w[5:9],1,axis=1)+3*z[:,:,None]**2*w[9:]\\n        dz=np.linalg.solve(J.transpose(0,2,1),(AF(z)@w-x)[...,None])[...,0]\\n        z-=np.clip(dz,-.4,.4)\\n    a=s[\'panel_b\']; b=B(z)@fit(B(a[\'u\']),a[\'v\']); a=s[\'panel_c\']; C=fit(D(P(a[\'u\'])),a[\'v1\']); W=fit(E(a[\'u\']),a[\'v2\']); e=E(z)@W\\n    return z,b,e,C,W\\n\\ndef kernel(x,n,degree,rbf,reg,y):\\n    x=x-x[:n].mean(0); x=x/max(.025,np.sqrt(np.mean(x[:n]**2))); L=x@x.T/x.shape[1]\\n    d=np.maximum(L.diagonal()[:,None]+L.diagonal()-2*L,0); K=(1+L)**degree\\n    if rbf: K+=np.exp(-rbf*d)\\n    return K[n:,:n]@np.linalg.solve(K[:n,:n]+reg*np.eye(n),y)\\n\\ndef solve_batch(study:dict)->dict:\\n    tr=study[\'train\']; n=len(tr[\'x\']); z,b,e,C,W=features(study,np.r_[tr[\'x\'],study[\'test\'][\'x\']]);\\n    options=_scalar_candidates(z,b,e,C,W,tr[\'y\'])\\n    y=0.5*(0.5*options[\'original\']+0.5*options[\'blend30\'])+0.5*_structured_y(z,b,C,tr[\'y\'],\'projsqp\',0.1)\\n    state=kernel(np.c_[D(P(.5*b))@C,.22*z],n,2,0,.0005,tr[\'traj\'])\\n    return {\'y_hat\':y,\'state_pred\':state,\'diagnostics_vec\':np.array([y.mean(),y.std(),state[:,-1].mean(),state[:,-1].std()])}\\n\\nimport numpy as np\\nfrom itertools import combinations_with_replacement\\nfrom scipy.linalg import cho_factor,cho_solve\\nfrom scipy.optimize import minimize, nnls\\n\\ndef _gram(a):\\n    return a@a.T/a.shape[1]\\n\\ndef _poly_parts(z,b,p):\\n    return [z,b,p]+[np.column_stack([v[:,i]*v[:,j] for i,j in combinations_with_replacement(range(v.shape[1]),2)]) for v in (z,b,p)]+[(u[:,:,None]*v[:,None,:]).reshape(len(z),-1) for u,v in ((z,b),(z,p),(b,p))]\\n\\ndef _scalar_candidates(z,b,e,C,W,y):\\n    n=len(y); p=P(b); c=D(p)@C; h=D(P(b+.5*e))@C; ep=E(b)@W\\n    candidates=[]\\n    cv=[]\\n    names=[]\\n    def add(K,key,reg,transform=0):\\n        K0=K[:n,:n].copy(); K0.flat[::n+1]+=reg\\n        ch=cho_factor(K0,check_finite=False)\\n        target=y if transform==0 else np.arcsinh(transform*y)/transform\\n        w=cho_solve(ch,target,check_finite=False); inv=np.diag(cho_solve(ch,np.eye(n),check_finite=False)); out=K[n:,:n]@w; lo=target-w/inv\\n        if transform: out=np.sinh(np.clip(out*transform,-4,4))/transform; lo=np.sinh(np.clip(lo*transform,-4,4))/transform\\n        candidates.append(out);cv.append(lo);names.append(key)\\n    def centered(x):return x-x[:n].mean(0)\\n    def gd(x):\\n        a=centered(x); a=a/max(.02,np.sqrt(np.mean(a[:n]**2))); L=_gram(a); d=np.maximum(L.diagonal()[:,None]+L.diagonal()-2*L,0); return L,d\\n    for s in (0,.2,.4,.6,.9):\\n      for v in (e,ep,z):\\n        q=D(P(b+s*v))@C\\n        for az in (.12,.22,.4,.65):\\n          for ab in (0,.35,.8):\\n            f=np.c_[az*z,ab*b,q]; L,d=gd(f)\\n            for k in (.12,.5,1.3):\\n              add((1+L)**2+np.exp(-k*d),(\'cascade\',s,az,ab,k),.00002)\\n    # Explicitly separate the linear, bilinear and quartic blocks rather than\\n    # shrinking each coefficient at the same rate.\\n    Q=_poly_parts(z,b,p); G=[]\\n    for q in Q:\\n        q=centered(q); G.append(_gram(q)/max(.0001,np.mean(q[:n]**2)))\\n    z0,b0,p0,zz,bb,pp,zb,zp,bp=G\\n    for w0 in (.08,.3,1.):\\n      for w1 in (.015,.1,.45,1.5):\\n       for w2 in (.01,.1,.5):\\n        for w3 in (0.,.025,.15):\\n         K=1+z0+w0*p0+w1*zp+w2*(zz+pp)+w3*(b0+bb+zb+bp)\\n         for rg in (.00015,.002,.014): add(K,(\'block\',w0,w1,w2,w3,rg),rg)\\n    for f in (np.c_[z,p],np.c_[z,p,b],np.c_[z,p,h],np.c_[z,p,ep]):\\n      for size in (2,3):\\n        X=np.column_stack([f]+[np.column_stack([np.prod(f[:,ids],axis=1) for ids in combinations_with_replacement(range(f.shape[1]),d)]) for d in range(2,size+1)])\\n        X=centered(X); X/=np.maximum(.02,np.sqrt(np.mean(X[:n]**2,0)))\\n        # Sparse empirical Bayes: diagonal coefficient variances with a stable noise floor.\\n        for floor in (.00008,.001,.004):\\n            V=X[:n]; w=np.full(X.shape[1],.025); residual=y-y.mean();\\n            for _ in range(240):\\n                K=(V*w)@V.T; K.flat[::n+1]+=floor\\n                ch=cho_factor(K,check_finite=False); A=cho_solve(ch,V,check_finite=False); mu=w*(A.T@residual); leverage=w*np.sum(V*A,0); nw=np.maximum(1e-8,mu*mu/np.maximum(.04,leverage)); nw=np.minimum(nw,4.); w=.64*w+.36*nw\\n            add(1+(X*w)@X.T,(\'sparse\',f.shape[1],size,floor),floor)\\n    L,d=gd(np.c_[h,.22*z,b]); add((1+L)**2+np.exp(-.7*d),(\'original\',),1e-6)\\n    pred=np.array(candidates); cv=np.array(cv); score=np.mean(abs(cv-y),axis=1); idx=np.argsort(score); out={}\\n    for family in (\'cascade\',\'block\',\'sparse\',\'original\'):\\n       ii=np.array([i for i in idx if names[i][0]==family]);\\n       out[family]=pred[ii[:min(12,len(ii))]].mean(0)\\n    for k in (8,30,100):out[\'blend\'+str(k)]=pred[idx[:k]].mean(0)\\n    # A convex, regularized ensemble of the best distinct leave-one-out predictors.\\n    ix=np.array([i for fam in (\'cascade\',\'block\',\'sparse\') for i in [j for j in idx if names[j][0]==fam][:15]]); V=cv[ix].T\\n    w=nnls(np.r_[V,np.full((1,len(ix)),30.),.035*np.eye(len(ix))],np.r_[y,30.,np.zeros(len(ix))])[0]; out[\'stack\']=w@pred[ix]/max(w.sum(),1e-8)\\n    return out\\n\\nfrom scipy.optimize import least_squares, minimize_scalar\\n\\ndef _structured_y(z,b,C,y,mode,pen):\\n    n=len(y); p=P(b); c=D(p)@C\\n    def scale(v):\\n        v=v-v[:n].mean(0)\\n        return v/np.maximum(.035,v[:n].std(0))\\n    z,p,c,b=map(scale,(z,p,c,b)); X=np.c_[z,p]; A=D(X); L=A[:n]; rng=np.random.default_rng(107)\\n    if mode in (\'rank1\',\'rank2\',\'fullrank1\',\'fullrank2\'):\\n        full=mode.startswith(\'full\'); d=8 if full else 4; rank=int(mode[-1]); U=X if full else z; V=X if full else p\\n        def prediction(q,A,U,V):\\n            a=q[9:9+d*rank].reshape(d,rank); h=q[9+d*rank:].reshape(d,rank)\\n            return A@q[:9]+np.sum((U@a)*(V@h),axis=1)\\n        def loss(q):return np.r_[prediction(q,L,U[:n],V[:n])-y,pen*q[9:]]\\n        def jac(q):\\n            a=q[9:9+d*rank].reshape(d,rank); h=q[9+d*rank:].reshape(d,rank)\\n            return np.r_[np.c_[L,(U[:n,:,None]*(V[:n]@h)[:,None,:]).reshape(n,-1),(V[:n,:,None]*(U[:n]@a)[:,None,:]).reshape(n,-1)],np.c_[np.zeros((2*d*rank,9)),pen*np.eye(2*d*rank)]]\\n        q0=fit(L,y); sol=min((least_squares(loss,np.r_[q0,rng.normal(0,.3,2*d*rank)],jac=jac,max_nfev=160) for _ in range(6)),key=lambda r:r.cost)\\n        return prediction(sol.x,A[n:],U[n:],V[n:])\\n    if mode.startswith(\'proj\'):\\n        base=D(np.c_[z,p]); kind=mode[4:]; V=p if kind[-1]==\'p\' else c; kind=kind[:-1]\\n        fs={\'sin\':np.sin,\'asinh\':np.arcsinh,\'tanh\':np.tanh,\'rat\':lambda x:x/(1+x*x),\'sq\':np.square}\\n        f=fs[kind]\\n        def build(q):\\n            h=f(q[0]*V+q[1]); h=scale(h); return np.c_[base,h,(z[:,:,None]*h[:,None,:]).reshape(len(z),-1)]\\n        def fitq(q,ret=False):\\n            H=build(q); F=H[:n]; ridge=np.r_[np.full(9,.004),np.full(4,pen*.5),np.full(16,pen)]; coef=np.linalg.solve(F.T@F+np.diag(ridge),F.T@y)\\n            return H[n:]@coef if ret else np.r_[F@coef-y,np.sqrt(ridge)*coef,.01*q]\\n        res=min((least_squares(fitq,[k,s],bounds=([.03,-3],[6,3]),max_nfev=65) for k in (.35,1.5,3.5) for s in (0.,.8)),key=lambda r:r.cost)\\n        return fitq(res.x,True)\\n    raise ValueError(mode)\\n"}'

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

def _collect_numeric_leaf_digests(obj) -> set[str]:
    out = set()
    def _walk(v):
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, (int, float, np.integer, np.floating)):
            out.add(_canon_digest(v))
        elif isinstance(v, np.ndarray):
            if v.size > 0 and v.dtype.kind in ('f', 'i', 'u'):
                out.add(_canon_digest(v))
        elif isinstance(v, dict):
            for sub in v.values():
                _walk(sub)
        elif isinstance(v, (list, tuple)):
            for sub in v:
                _walk(sub)
    _walk(obj)
    return out

def _collect_subtree_digests(obj) -> set[str]:
    out = {_canon_digest(obj)}
    if isinstance(obj, dict):
        for sub in obj.values():
            out.update(_collect_subtree_digests(sub))
    elif isinstance(obj, (list, tuple)):
        for sub in obj:
            out.update(_collect_subtree_digests(sub))
    return out

class _TrackedPayloadDict(dict):
    def __init__(self, data: dict, src_id: str, consumed_set: set[str]):
        super().__init__(data)
        self._src_id = src_id
        self._consumed_set = consumed_set
    def __getitem__(self, key):
        val = super().__getitem__(key)
        self._consumed_set.add(self._src_id)
        return val
    def get(self, key, default=None):
        if super().__contains__(key):
            self._consumed_set.add(self._src_id)
        return super().get(key, default)
    def values(self):
        if len(self) > 0:
            self._consumed_set.add(self._src_id)
        return super().values()
    def items(self):
        if len(self) > 0:
            self._consumed_set.add(self._src_id)
        return super().items()

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
    raw_input_copy = deepcopy(study)
    raw_input_dig = _canon_digest(raw_input_copy)
    outputs = {'__input__': raw_input_copy}
    exec_counts = {nid: 0 for nid in _SPEC_OBJ['topo_order']}
    non_identity_checks = {}
    edge_consumption_checks = {}
    is_multi_node = len(_SPEC_OBJ['topo_order']) > 1
    for nid in _SPEC_OBJ['topo_order']:
        srcs = _SPEC_OBJ['incoming'][nid]
        consumed_srcs: set[str] = set()
        if not is_multi_node and srcs == ['__input__']:
            node_in = _TrackedPayloadDict(deepcopy(outputs['__input__']), '__input__', consumed_srcs)
        else:
            node_in = {
                s: (_TrackedPayloadDict(deepcopy(outputs[s]), s, consumed_srcs) if isinstance(outputs[s], dict) else deepcopy(outputs[s]))
                for s in srcs
            }
        in_dig = _canon_digest(outputs['__input__'] if (not is_multi_node and srcs == ['__input__']) else {s: outputs[s] for s in srcs})
        in_num_leaves = set()
        for s in srcs:
            in_num_leaves.update(_collect_numeric_leaf_digests(outputs[s]))
        consumed_srcs.clear()
        out_v = _COMPILED_FNS[nid](node_in)
        exec_counts[nid] += 1
        if exec_counts[nid] != 1:
            raise RuntimeError(f'DAG node {nid} executed {exec_counts[nid]} times != 1')
        if not isinstance(out_v, dict) or not out_v:
            raise RuntimeError(f'DAG node {nid} must return a non-empty dict')
        if is_multi_node:
            unused_srcs = sorted(set(srcs) - consumed_srcs)
            if unused_srcs:
                raise RuntimeError(f'DAG node {nid} left declared incoming edges unconsumed: {unused_srcs}')
            edge_consumption_checks[nid] = sorted(consumed_srcs)
        else:
            edge_consumption_checks[nid] = sorted(consumed_srcs) if consumed_srcs else list(srcs)
        out_dig = _canon_digest(out_v)
        if nid != _SPEC_OBJ['sink_node_id']:
            # Strict anti-wrapper-shell & genuine numeric transformation check (`U06`):
            out_subtrees = _collect_subtree_digests(out_v)
            echoed = (out_dig == in_dig) or (raw_input_dig in out_subtrees) or any(out_dig == _canon_digest(outputs[s]) for s in srcs)
            out_num_leaves = _collect_numeric_leaf_digests(out_v)
            novel_num_leaves = out_num_leaves - in_num_leaves
            if echoed or not out_num_leaves or not novel_num_leaves:
                raise RuntimeError(f'DAG node {nid} failed anti-wrapper-shell / novel numeric transformation check')
            non_identity_checks[nid] = True
        # Convert any tracked dict views returned by user code into plain dicts before freezing in outputs
        outputs[nid] = {k: (dict(v) if isinstance(v, _TrackedPayloadDict) else v) for k, v in out_v.items()}
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
            'edge_consumption_checks': edge_consumption_checks,
            'all_declared_edges_consumed': all(set(edge_consumption_checks.get(n, [])) == set(_SPEC_OBJ['incoming'][n]) for n in _SPEC_OBJ['topo_order']) if is_multi_node else True,
            'node_code_shas': _SPEC_OBJ['node_code_shas'],
        }
        open(trace_fp, 'w', encoding='utf-8').write(json.dumps(trace_obj, sort_keys=True))
    return outputs[_SPEC_OBJ['sink_node_id']]
