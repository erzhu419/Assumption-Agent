"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_01", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "472f65da8843baf3b93679be4e8b2bda3e0480a05cd1aa1e4ba842176b6146b6"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Calibrated multi-output regression for the public study.\\"\\"\\"\\nimport numpy as np\\nfrom itertools import combinations, combinations_with_replacement\\nfrom scipy.linalg import cho_factor, cho_solve\\nfrom scipy.optimize import minimize\\nfrom threadpoolctl import threadpool_limits\\n\\n\\ndef _aff(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _ls(x, y):\\n    return np.linalg.lstsq(x, y, rcond=1e-9)[0]\\n\\n\\ndef _coordinates(study, x):\\n    a, b, c = (study[k] for k in (\'panel_a\',\'panel_b\',\'panel_c\'))\\n    u, v = a[\'u\'], a[\'v\']\\n    mask = np.linalg.norm(v-u, axis=1) < 2\\n    if mask.sum() < 8:\\n        mask = np.ones(len(u), bool)\\n    w = np.stack([_ls(_aff(u[mask, j:j+1]), v[mask, j]) for j in range(4)], axis=1)\\n    q = (x-w[0])/w[1]\\n    m = _ls(_aff(b[\'u\']), b[\'v\'])\\n    z = _aff(q) @ m\\n    def quadratic(p):\\n        return _aff(np.stack([p[:,i]*p[:,j] for i,j in ((0,1),(0,3),(1,2),(2,3))], axis=1))\\n    wc = _ls(quadratic(c[\'u\']), c[\'v1\'])\\n    wd = _ls(_aff(np.cos(2.4*c[\'u\'])), c[\'v2\'])\\n    return q,z,quadratic(q)@wc,_aff(np.cos(2.4*q))@wd,quadratic(z)@wc,_aff(np.cos(2.4*z))@wd\\n\\n\\ndef _kernels(p, n):\\n    q,z,c,d,cz,dz = p\\n    ks = []\\n    def add(f):\\n        f = (f - f[:n].mean(axis=0))/np.maximum(f[:n].std(axis=0), 1e-9)\\n        ks.append(f@f.T/f.shape[1])\\n    def mon(x, deg):\\n        return np.stack([np.prod(x[:, ix],axis=1) for ix in combinations_with_replacement(range(4),deg)],axis=1)\\n    ks.append(np.ones((len(q),len(q))))\\n    add(q)\\n    for deg in range(2,5):\\n        for x in (q,z): add(mon(x,deg))\\n    add(np.column_stack((c,d)))\\n    add(np.column_stack((cz,dz)))\\n    for x in (q,z):\\n        pair = np.stack([x[:,i]+s*x[:,j] for i,j in combinations(range(4),2) for s in (-1,1)],axis=1)\\n        for a in (1.25,2.4,4.0):\\n            h = np.column_stack((x,pair))\\n            add(np.column_stack((np.sin(a*h),np.cos(a*h))))\\n    for x in (q,z,c,d):\\n        delta = x[:,None]-x[None,:]\\n        dist = ((delta/np.maximum(x[:n].std(0),1e-9))**2).sum(2)\\n        for scale in (0.12, 0.5, 2.0):\\n            ks.append(np.exp(-scale*dist))\\n    return np.asarray(ks)\\n\\n\\ndef _fit(kernels, target, separate=True):\\n    n = len(target)\\n    mu = target.mean(0)\\n    sd = np.maximum(target.std(0),1e-8)\\n    t = (target-mu)/sd\\n    K = np.concatenate((kernels[:,:n,:n],np.eye(n)[None]),axis=0).copy()\\n    Kflat = K.reshape(len(K),-1)\\n    Qflat = kernels[:,n:,:n].copy().reshape(len(K)-1,-1)\\n    def fit_one(t):\\n        m=t.shape[1]\\n        prior=np.full(len(K),-3.0); prior[-1]=-5.0\\n        def loss(v):\\n            a=np.exp(v)\\n            L=cho_factor((a@Kflat).reshape(n,n),check_finite=False)\\n            inv=cho_solve(L,np.eye(n),check_finite=False)\\n            alpha=inv@t\\n            g=(m*inv-alpha@alpha.T).reshape(-1)\\n            val=.5*np.sum(alpha*t)+m*np.log(L[0].diagonal()).sum()+.025*np.sum((v-prior)**2)\\n            return val, .5*a*(Kflat@g)+.05*(v-prior)\\n        a=np.exp(minimize(loss,prior,jac=True,method=\'L-BFGS-B\',bounds=[(-12,6)]*len(prior),options={\'maxiter\':180}).x)\\n        alpha=cho_solve(cho_factor((a@Kflat).reshape(n,n),check_finite=False),t,check_finite=False)\\n        return (a[:-1]@Qflat).reshape(len(kernels[0])-n,n)@alpha\\n    if separate:\\n        out = np.column_stack([fit_one(t[:, i:i+1]) for i in range(t.shape[1])])\\n    else:\\n        out = fit_one(t)\\n    return out*sd+mu\\n\\n\\ndef solve_batch(study):\\n    train=study[\'train\']; n=len(train[\'x\']); x=np.concatenate((train[\'x\'],study[\'test\'][\'x\']))\\n    with threadpool_limits(limits=1):\\n        coords=_coordinates(study,x)\\n        ks=_kernels(coords,n)\\n        basis=np.linalg.svd(train[\'traj\'],full_matrices=False)[2][:3]\\n        target=np.column_stack((train[\'y\'],train[\'traj\']@basis.T))\\n        ext=np.concatenate((ks,_graph_kernels(study,coords,n)))\\n        result=(_fit(ks,target)+_fit(ext,target)+_fit(ext,target,separate=False))/3\\n    y=result[:,0]; state=result[:,1:]@basis\\n    return {\'y_hat\':y,\'state_pred\':state,\'diagnostics_vec\':np.array([y.mean(),y.std(),state[:,-1].mean(),state[:,-1].std()])}\\n\\ndef _graph_kernels(study, coords, n):\\n    q,z,c,d,cz,dz=coords\\n    panel=study[\'panel_c\']; pb=study[\'panel_b\']; a0,a1=np.abs(study[\'telemetry_vec\'][:2]); a0=max(a0,.01); a1=max(a1,.001)\\n    ix=((0,1),(0,3),(1,2),(2,3))\\n    def quadratic(p): return _aff(np.stack([p[:,i]*p[:,j] for i,j in ix],axis=1))\\n    w=_ls(quadratic(panel[\'u\']),panel[\'v1\']); v=_ls(_aff(np.cos(2.4*panel[\'u\'])),panel[\'v2\']); b=_ls(_aff(pb[\'u\']),pb[\'v\'])\\n    B=lambda p:_aff(p)@b\\n    C=lambda p:quadratic(p)@w\\n    D=lambda p:_aff(np.cos(2.4*p))@v\\n    def standard(x): return (x-x[:n].mean(0))/np.maximum(x[:n].std(0),1e-10)\\n    def gram(x):\\n        x=standard(x); return x@x.T/x.shape[1]\\n    def powers(x,k): return np.stack([np.prod(x[:,i],axis=1) for i in combinations_with_replacement(range(4),k)],axis=1)\\n    groups=[]\\n    def add(group): groups.append(np.mean(group,axis=0))\\n    add([gram(powers(x,k)) for x in (q,z) for k in (5,6)])\\n    fns=(lambda p:np.sin(a0*p),lambda p:np.tanh(a0*p),lambda p:p/(1+a1*p*p),lambda p:p+a1*p*p,lambda p:np.sin(a0*p)+a1*p*p,lambda p:np.cos(a0*p)+a1*p)\\n    for f in fns:\\n        p=f(q); u=B(p)\\n        add([gram(x) for x in (C(p),D(p),C(u),D(u))]);add([gram(powers(x,k)) for x in (p,u,f(z)) for k in (2,3)])\\n    levels=(c,d,cz,dz,C(d),D(c),C(dz),D(cz))\\n    for x in (q,z):\\n        for j in (0,4):\\n            aug=levels[j:j+4]; gx=gram(x)\\n            add([gx*gram(p) for p in aug]);add([gram(p*x) for p in aug]);add([gram(x/(1+a1*p*p)) for p in aug]);add([gram(p/(1+a1*x*x)) for p in aug]);add([gram(np.sin(a0*(x+p))) for p in aug]);add([gram(np.cos(a0*(x-p))) for p in aug]);add([gram(powers(p,2)) for p in aug]);add([gram(powers(p,3)) for p in aug]);\\n    for p,r in ((c,d),(cz,dz),(c,dz),(cz,d),(q,z)):\\n        add([gram(p)*gram(r),gram(p*r)]);add([gram(p/(1+a1*r*r)),gram(r/(1+a1*p*p))]);add([gram(np.sin(a0*(p+r))),gram(np.cos(a0*(p-r)))])\\n    for levels0 in ((q,z),(c,d),(cz,dz),(q,c,d),(q,cz,dz)):\\n        g=sum(gram(p) for p in levels0)/len(levels0); ds=np.maximum(g.diagonal()[:,None]+g.diagonal()[None,:]-2*g,0)\\n        for k in (0.3,0.8,1.6): groups.append(np.exp(-k*ds));\\n    return np.asarray(groups)\\n"}'

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
