"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_10", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "db8b3a778d231a3baffedb98b4fd14f980654505c5f5e9067a0ff95dc6adda7e"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Panel-calibrated scalar and trajectory prediction.\\"\\"\\"\\nimport numpy as np\\nfrom itertools import combinations_with_replacement\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _ls(a, b):\\n    return np.linalg.lstsq(a, b, rcond=1e-11)[0]\\n\\n\\ndef _roll(x):\\n    return np.roll(x, -1, axis=1)\\n\\n\\ndef _coords(study, x):\\n    u, v = [np.asarray(study[\'panel_a\'][k], float) for k in (\'u\',\'v\')]\\n    good = np.linalg.norm(v, axis=1) < 5\\n    u, v = u[good], v[good]\\n    def features(z):\\n        return np.stack((np.ones_like(z), z, _roll(z), z*_roll(z), z**3), axis=2)\\n    a = features(u)\\n    c = np.stack([_ls(a[:, j], v[:, j]) for j in range(4)], axis=1)\\n    z = np.array(x, dtype=float, copy=True)\\n    for _ in range(28):\\n        delta = np.sum(features(z)*c.T, axis=2) - x\\n        jac = np.eye(4)[None,:,:]*(c[1]+c[3]*_roll(z)+3*c[4]*z*z)[:,:,None]\\n        jac += np.roll(np.eye(4), 1, axis=1)[None,:,:]*(c[2]+c[3]*z)[:,:,None]\\n        z -= np.clip(np.linalg.solve(jac, delta[...,None])[...,0], -.5, .5)\\n    return z\\n\\n\\ndef _poly(h, degree):\\n    return np.column_stack([np.ones(len(h))] + [np.prod(h[:, idx], axis=1) for d in range(1,degree+1) for idx in combinations_with_replacement(range(h.shape[1]),d)])\\n\\n\\ndef _panel_b(study, z):\\n    u, v = [np.asarray(study[\'panel_b\'][k], float) for k in (\'u\',\'v\')]\\n    def f(x, w):\\n        return np.c_[np.ones(len(x)), x, np.sin(w*x)]\\n    w = least_squares(lambda w: (f(u,w)@_ls(f(u,w),v)-v).ravel(), [2.9], bounds=(1., 4.)).x\\n    return f(z,w)@_ls(f(u,w),v)\\n\\n\\nfrom itertools import combinations\\nfrom scipy.linalg import cho_factor, cho_solve\\n\\n\\ndef _readout_features(study, z, n):\\n    groups = {}\\n    def add(name, a):\\n        a = np.array(a, dtype=float, copy=True).reshape(len(z), -1)\\n        a -= a[:n].mean(0)\\n        a /= np.maximum(np.sqrt(np.mean(a[:n] ** 2, axis=0)), 1e-9)\\n        groups[name] = a\\n    def expand(name, a):\\n        add(name+\'L\', a)\\n        add(name+\'Q\', a*a)\\n        add(name+\'X\', np.column_stack([a[:,i]*a[:,j] for i,j in combinations(range(4),2)]))\\n    def cross(a,b):\\n        return (a[:,:,None]*b[:,None,:]).reshape(len(z),-1)\\n    h0 = _panel_b(study, z)\\n    h = h0-h0[:n].mean(0)\\n    expand(\'H\',h)\\n    expand(\'Z\',z)\\n    add(\'HZ\',cross(h,z))\\n    coeff = _ls(np.c_[np.ones(n),z[:n]],h[:n]); r = h-coeff[0]-z@coeff[1:]\\n    r = r@np.linalg.svd(r[:n],full_matrices=False)[2].T\\n    expand(\'R\',r); add(\'RZ\',cross(r,z))\\n    for tag, a in [(\'H\',h),(\'Z\',z)]:\\n        add(tag+\'D3\',_poly(a,3)[:,15:])\\n        add(tag+\'T\',np.column_stack([np.prod(a[:,i],axis=1) for i in combinations(range(4),3)]))\\n    # Compact auxiliary operator bases estimated without any downstream labels.\\n    for key in (\'v1\',\'v2\'):\\n        u,v = [study[\'panel_c\'][k] for k in (\'u\',key)]\\n        if key==\'v1\':\\n            def f(x): return np.c_[np.ones(len(x)),x,np.arctan(2.43*x)]\\n        else:\\n            def f(x): return np.c_[np.ones(len(x)),x,x*x,np.cos(2.569*x)]\\n        c=_ls(f(u),v)\\n        a=f(z)@c; b=f(h0)@c\\n        a-=a[:n].mean(0); b-=b[:n].mean(0)\\n        for suffix, val in [(\'Z\',a),(\'H\',b)]:\\n            tag=key+suffix\\n            expand(tag,val); add(tag+\'joint\',cross(val,h))\\n    for tag,g in [(\'rat\',z**3/(1+.65*z*z)),(\'sin\',z-.4*np.sin(2.89*z)),(\'soft\',np.tanh(1.25*z))]:\\n        expand(tag,g); add(tag+\'mix\',cross(z,g))\\n    # Mixed-basis polynomial kernels keep higher-order interactions shrunk as a group.\\n    for tag,a in [(\'A\',np.c_[groups[\'HL\'],groups[\'ZL\']]),(\'B\',groups[\'HL\']),(\'C\',np.c_[groups[\'HL\'],groups[\'v1ZL\'],groups[\'v2ZL\']])]:\\n        for power in (2,3,4):\\n            a0=a/np.sqrt(a.shape[1]); add(tag+\'p\'+str(power),_poly(a0,power)[:,_poly(a0,power-1).shape[1]:])\\n    return groups\\n\\n\\ndef _stable_predict(study, z, Y, n):\\n    \\"\\"\\"Fit smooth readouts with explicit native/operator interactions and blocked validation.\\"\\"\\"\\n    groups = _readout_features(study, z, n)\\n    kernel = {k: a @ a.T / a.shape[1] for k, a in groups.items()}\\n    z0 = groups[\'ZL\']; h0 = groups[\'HL\']\\n    v = study[\'panel_b\'][\'v\']; u = study[\'panel_b\'][\'u\']\\n    # Recover the componentwise near-odd source in panel B, with an optional\\n    # smooth residual correction tied to the panel measurements.\\n    def base(u, w):\\n        return np.c_[np.ones(len(u)), u, np.sin(w*u)]\\n    def panel_res(w):\\n        F = base(u,w); return (F@_ls(F,v)-v).ravel()\\n    w = least_squares(panel_res,[2.9],bounds=(1.,4.)).x\\n    coef = _ls(base(u,w),v)\\n    residual = v-base(u,w)@coef\\n    for scale in (.25,.8):\\n        d1 = ((u[:,None,:]-u[None,:,:])**2).sum(2)\\n        d2 = ((z[:,None,:]-u[None,:,:])**2).sum(2)\\n        a = base(z,w)@coef + np.exp(-scale*d2)@np.linalg.solve(np.exp(-scale*d1)+.04*np.eye(len(u)),residual)\\n        a -= a[:n].mean(0); a /= np.maximum(a[:n].std(0), 1e-8)\\n        dot = a@a.T/4\\n        kernel[\'bc\'+str(scale)] = dot + dot**2 + dot*(z0@z0.T/4)\\n    H = sum(kernel[k] for k in (\'HL\',\'HQ\',\'HX\'))\\n    Z = sum(kernel[k] for k in (\'ZL\',\'ZQ\',\'ZX\'))\\n    M = kernel[\'HZ\']\\n    D = sum(kernel[k] for k in (\'v1ZL\',\'v1ZQ\',\'v1ZX\',\'v2ZL\',\'v2ZQ\',\'v2ZX\'))\\n    S = sum(kernel[k] for k in (\'sinL\',\'sinQ\',\'sinX\',\'sinmix\'))\\n    R = sum(kernel[k] for k in (\'ratL\',\'ratQ\',\'ratX\',\'ratmix\'))\\n    C = sum(kernel[k] for k in (\'v1HL\',\'v1HQ\',\'v1HX\',\'v2HL\',\'v2HQ\',\'v2HX\'))\\n    # A single observed coordinate may be relevant in both even and odd\\n    # parity channels. Keep cross-coordinate degrees separate.\\n    def sparse_tensor(a):\\n        x = a / np.maximum(a[:n].std(0), 1e-9)\\n        k = x @ x.T / x.shape[1]\\n        d = ((x[:,None,:]-x[None,:,:])**2).sum(2)/x.shape[1]\\n        return k, d\\n    hk,hd = sparse_tensor(h0); zk,zd = sparse_tensor(z0)\\n    families = []\\n    families.append((\'native\',H+Z+M))\\n    for w0 in (.08,.35,1.,3.):\\n        B=H+w0*(Z+M)\\n        families.extend([(\'mix\',B),(\'rat\',B+w0*R),(\'sin\',B+w0*S),(\'aux\',B+w0*(D+C)),(\'panel\',B+w0*(kernel[\'bc0.25\']+kernel[\'bc0.8\'])),(\'branch\',B+w0*(kernel[\'v1Zjoint\']+kernel[\'v2Zjoint\']+kernel[\'v1Hjoint\']+kernel[\'v2Hjoint\']))])\\n        for w1 in (.025,.1):\\n            families.append((\'poly\',B+w1*(hk**3+zk**3+(hk+zk)**3/8)))\\n        for gamma in (.1,.35,.9):\\n            families.append((\'kernel\',B+.35*np.exp(-gamma*hd)+w0*np.exp(-gamma*zd)+w0*np.exp(-gamma*(hd+zd)/2)))\\n    for w0 in (.1,.5,2.):\\n        families.append((\'interaction\',H+.2*Z+w0*(kernel[\'RZ\']+kernel[\'RQ\']+kernel[\'RX\'])))\\n    # Random folds plus coordinate-tail blocks penalize unstable extrapolation.\\n    rng = np.random.default_rng(326)\\n    ids = np.arange(n)\\n    folds = []\\n    for _ in range(2):\\n        for val in np.array_split(rng.permutation(n),4):\\n            folds.append((np.setdiff1d(ids,val),val,1.))\\n    for j in range(z.shape[1]):\\n        order=np.argsort(z[:n,j]); val=np.r_[order[:n//9],order[-(n//9):]]\\n        folds.append((np.setdiff1d(ids,val),val,.45))\\n    def predict(K,reg,fit,ev):\\n        A=K[np.ix_(fit,fit)] + reg*np.eye(len(fit))\\n        cf=cho_factor(A,check_finite=False)\\n        return K[np.ix_(ev,fit)]@cho_solve(cf,Y[fit],check_finite=False)\\n    rows=[]; ev=np.arange(n,len(z)); regs=(.00004,.00015,.0005,.0015,.004,.01,.025)\\n    for name,K0 in families:\\n        K=K0+1\\n        for reg in regs:\\n            err=np.zeros(2); weight=0.\\n            for fit,val,fw in folds:\\n                e=abs(predict(K,reg,fit,val)-Y[val]); err+=fw*np.array([e[:,0].mean(),e[:,1:].mean()]);weight+=fw\\n            rows.append((err/weight,name,predict(K,reg,ids,ev)))\\n    P=[]\\n    for i in (0,1):\\n        rows.sort(key=lambda r:r[0][i]); seen=set(); best=[]\\n        for e,name,p in rows:\\n            if name in seen: continue\\n            seen.add(name); best.append((e[i],p))\\n            if len(best)>=5:break\\n        e=np.array([r[0] for r in best]); W=np.exp(-(e-e.min())/(.08*max(e.min(),1e-6))); W/=W.sum()\\n        P.append(np.tensordot(W,np.array([r[1] for r in best]),axes=1))\\n    return np.column_stack((P[0][:,0],P[1][:,1:]))\\n\\n\\ndef solve_batch(study):\\n    tr, te = study[\'train\'], study[\'test\']\\n    n=len(tr[\'x\']); z=_coords(study,np.concatenate((tr[\'x\'],te[\'x\']),axis=0))\\n    target=np.column_stack((tr[\'y\'],tr[\'traj\']))\\n    pred=_stable_predict(study,z,target,n)\\n    y,t=pred[:,0],pred[:,1:]\\n    return {\'y_hat\':y,\'state_pred\':t,\'diagnostics_vec\':np.array([y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std()])}\\n"}'

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
