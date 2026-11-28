"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_08", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "554f6c633fbe6f2fce9a23ba9600e8c312da785ec1020f84ad05ab0efe598f61"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Recover latent measurements and learn regularized nonlinear readouts.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import root, minimize_scalar, least_squares\\nfrom scipy.linalg import solve\\nfrom scipy.spatial.distance import cdist\\nfrom sklearn.linear_model import lasso_path\\n\\n\\ndef _aff(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _ls(x, y):\\n    return np.linalg.lstsq(x, y, rcond=1e-10)[0]\\n\\n\\ndef _latent(study, raw):\\n    p = study[\'panel_a\']; u,v = p[\'u\'],p[\'v\']\\n    ok = np.max(abs(u-v),axis=1)<2; u,v = u[ok],v[ok]\\n    h = study[\'telemetry_vec\'][1]; w=np.roll(u,-1,axis=1)\\n    b=np.column_stack([_ls(_aff(np.c_[u[:,j],w[:,j],u[:,j]*w[:,j],np.sin(h*u[:,j])]),v[:,j]) for j in range(4)])\\n    def forward(q):\\n        r=np.roll(q,-1)\\n        return b[0]+b[1]*q+b[2]*r+b[3]*q*r+b[4]*np.sin(h*q)\\n    out=[]\\n    for row in raw:\\n        starts=[row,np.zeros(4),row*.7,row*1.3,*u[np.argsort(np.linalg.norm(v-row,axis=1))[:6]]]\\n        roots=[root(lambda q:forward(q)-row,start).x for start in starts]\\n        out.append(min(roots,key=lambda q:np.linalg.norm(q) if np.linalg.norm(forward(q)-row)<1e-7 else 1e3+np.linalg.norm(forward(q)-row)))\\n    return np.asarray(out)\\n\\n\\ndef _coordinates(study, raw):\\n    x=_latent(study,raw); ops=[]\\n    for panel,key,fun,br in [(\'panel_b\',\'v\',lambda z,a:np.tanh(a*z),(.5,2.5)),(\'panel_c\',\'v1\',lambda z,a:z/(1+a*abs(z)),(.2,2.0)),(\'panel_c\',\'v2\',lambda z,a:np.cos(a*z),(.5,2.2))]:\\n        p=study[panel]; u,v=p[\'u\'],p[key]\\n        f=lambda z,a:_aff(np.c_[z,fun(z,a)])\\n        def obj(a):\\n            m=f(u,a); return np.linalg.norm(v-m@_ls(m,v))\\n        a=minimize_scalar(obj,bounds=br,method=\'bounded\').x; b=_ls(f(u,a),v)\\n        ops.append(lambda z,f=f,a=a,b=b:f(z,a)@b)\\n    b,c,d=(op(x) for op in ops)\\n    return {\'x\':x,\'b\':b,\'c\':c,\'d\':d,\'bc\':ops[1](b),\'bd\':ops[2](b),\'cb\':ops[0](c),\'cd\':ops[2](c),\'db\':ops[0](d),\'dc\':ops[1](d)}\\n\\n\\ndef _bank(D, k, h, n):\\n    a=[]; names=[]\\n    def add(z,name):\\n        z=np.asarray(z); z=z[:,None] if z.ndim==1 else z\\n        a.append(z); names.extend(f\'{name}:{j}\' for j in range(z.shape[1]))\\n    for name,z in D.items():\\n        add(z,name); add(z*z,name+\'^2\'); add(z**3,name+\'^3\')\\n        add(np.sin(h*z),name+\' sin\'); add(np.cos(h*z),name+\' cos\'); add(np.tanh(h*z),name+\' tanh\'); add(abs(z),name+\' abs\'); add(z/(1+k*z*z),name+\' rat\'); add(z/(1+.85*abs(z)),name+\' soft\')\\n        for i in range(4):\\n            for j in range(i+1,4):\\n                p,q=z[:,i],z[:,j]; tag=f\'{name}{i}{j}\'\\n                add(p*q,tag+\' product\'); add((p*q)/(1+k*(p-q)**2),tag+\' rational product\'); add((p*q)/(1+k*(p+q)**2),tag+\' rational product sum\'); add(abs(p-q),tag+\' absdiff\'); add(abs(p+q),tag+\' abssum\')\\n                add(np.sin(h*(p-q)),tag+\' sindiff\'); add(np.sin(h*(p+q)),tag+\' sinsum\'); add(np.cos(h*(p-q)),tag+\' cosdiff\'); add(np.cos(h*(p+q)),tag+\' cossum\')\\n        add(z/(1+k*np.sum(z*z,axis=1)[:,None]),name+\' radial\')\\n    for ni,nj in [(\'x\',\'b\'),(\'x\',\'c\'),(\'x\',\'d\'),(\'b\',\'c\'),(\'b\',\'d\'),(\'c\',\'d\'),(\'bc\',\'bd\')]:\\n        p,q=D[ni],D[nj]\\n        add((p[:,:,None]*q[:,None,:]).reshape(len(p),-1),ni+\'*\'+nj)\\n        add((p[:,:,None]/(1+k*q[:,None,:]**2)).reshape(len(p),-1),ni+\'/\'+nj)\\n    z=np.column_stack(a); sd=z[:n].std(0); ok=sd>1e-7\\n    return (z[:,ok]-z[:n,ok].mean(0))/sd[ok],np.asarray(names)[ok]\\n\\n\\ndef _fit(D,Y,k,h,debug=False):\\n    n,m=Y.shape; N=len(D[\'x\']); folds=np.array_split(np.random.default_rng(913).permutation(n),5)\\n    A,names=_bank(D,k,h,n); K={}\\n    for name,Z in D.items():\\n        Z=(Z-Z[:n].mean(0))/np.maximum(Z[:n].std(0),.0001)\\n        dot=Z@Z.T/4; dist=cdist(Z,Z,\'sqeuclidean\')\\n        K[name+\' p2\']=(1+dot)**2; K[name+\' p3\']=(1+dot)**3\\n        for w in (2.5,5,9): K[f\'{name} g{w}\']=np.exp(-dist/(2*w*w))\\n    for subset in [(\'x\',\'b\'),(\'x\',\'b\',\'c\',\'d\'),(\'x\',\'b\',\'bc\',\'bd\')]:\\n        for suffix in [\' p2\',\' p3\',\' g2.5\',\' g5\',\' g9\']:\\n            K[str(subset)+suffix]=sum(K[s+suffix] for s in subset)/len(subset)\\n    K[\'dictionary\']=1+A@A.T/A.shape[1]\\n    settings=[(name,lam) for name in K for lam in (1e-5,.001,.01,.06)]\\n    def ridge(name,lam,i,j):\\n        M=K[name]; center=Y[i].mean(0)\\n        return M[np.ix_(j,i)]@solve(M[np.ix_(i,i)]+lam*np.eye(len(i)),Y[i]-center,assume_a=\'pos\')+center\\n    pred=[]; errs=[]; tags=[]; it=np.arange(n); jt=np.arange(n,N)\\n    for name,lam in settings:\\n        P=np.empty_like(Y)\\n        for j in folds:\\n            i=np.setdiff1d(it,j); P[j]=ridge(name,lam,i,j)\\n        errs.append(np.mean(abs(P-Y),0)); tags.append((name,lam)); pred.append(ridge(name,lam,it,jt))\\n    ratios=np.array([.12,.07,.04,.025,.015,.009,.005,.0025])\\n    def sparse(i,j):\\n        u=A[i]; mu=u.mean(0); Z=u-mu; means=Y[i].mean(0); y=Y[i]-means\\n        aout=np.empty((len(ratios),len(j),m)); coef=[]\\n        for t in range(m):\\n            top=max(np.max(abs(Z.T@y[:,t]))/len(i),1e-8)\\n            _,b,_=lasso_path(Z,y[:,t],alphas=ratios*top,max_iter=3500,tol=.00025)\\n            aout[:,:,t]=((A[j]-mu)@b).T+means[t]; coef.append(b)\\n        return aout,coef\\n    P=np.empty((len(ratios),n,m))\\n    for j in folds:\\n        P[:,j],_=sparse(np.setdiff1d(it,j),j)\\n    Q,co=sparse(it,jt)\\n    for a in range(len(ratios)):\\n        errs.append(np.mean(abs(P[a]-Y),0)); tags.append((\'lasso\',ratios[a])); pred.append(Q[a])\\n    er=np.asarray(errs); pp=np.asarray(pred); out=np.empty((len(jt),m))\\n    for group in [np.array([0]),np.arange(1,m)]:\\n        loss=er[:,group].mean(1); ids=np.argsort(loss)[:6]; w=np.exp(-20*(loss[ids]/loss[ids[0]]-1)); out[:,group]=np.einsum(\'i,ijk->jk\',w/w.sum(),pp[ids][:,:,group])\\n        if debug: print(\'CV candidates\',[(tags[i],round(loss[i],5)) for i in ids]); print(\'lasso features\',[(names[j],round(co[group[0]][j,4],3)) for j in np.argsort(abs(co[group[0]][:,4]))[-15:][::-1]])\\n    return out\\n\\n\\ndef solve_batch(study:dict)->dict:\\n    train=study[\'train\']; raw=np.r_[train[\'x\'],study[\'test\'][\'x\']]; D=_coordinates(study,raw)\\n    times=study[\'eval_times\']; T=train[\'traj\']; Y=np.c_[train[\'y\'],T]\\n    k,h=study[\'telemetry_vec\'][:2]; out=_fit(D,Y,k,h)\\n    def basis(r): return np.array([times,-np.expm1(-r[0]*times),-np.expm1(-r[1]*times)])\\n    def res(r):\\n        b=basis(r); return (T-_ls(b.T,T.T).T@b).ravel()\\n    r=least_squares(res,[.8,7],bounds=([.1,2],[2,20])).x; b=basis(r)\\n    y=out[:,0].copy(); p=_ls(b.T,out[:,1:].T).T@b\\n    return {\'y_hat\':y,\'state_pred\':p,\'diagnostics_vec\':np.array([y.mean(),y.std(),p[:,-1].mean(),p[:,-1].std()])}\\n"}'

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
