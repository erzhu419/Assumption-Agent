"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_10", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "284f1b68663f50b65918a2b6fd0c791e2564f2e5cedc59042ecf656eae024ccf", "node_01": "98d65092b7379a4648e553322eb1f55312f0c0b3688603354fd848360ba0581d"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Recover calibrated coordinates and evaluate the two response branches.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares, minimize_scalar\\n\\n\\ndef _ones(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _ls(a, y):\\n    return np.linalg.lstsq(a, y, rcond=None)[0]\\n\\n\\ndef _align(p, x):\\n    u, v = p[\'u\'], p[\'v\']\\n    # Gross contamination in the reference readings is shared across channels.\\n    keep = np.max(abs(v), axis=1) < 3.5\\n    u, v = u[keep], v[keep]\\n    best = None\\n    for typ in (\'sin\',\'tanh\',\'rat\'):\\n        def g(a):\\n            if typ == \'sin\': return np.sin(a)\\n            if typ == \'tanh\': return np.tanh(a)\\n            return a/(1+a*a)\\n        def basis(w, a): return _ones(np.c_[w,g(a[0]*w),g(a[1]*np.roll(w,2,axis=1))])\\n        def coeff(w,a,y):\\n            B=basis(w,a); return np.stack([_ls(B[:,[0,1+j,1+(j+2)%4,5+j,9+j]], y[:,j]) for j in range(4)])\\n        def calc(w,a,c):\\n            B=basis(w,a); return np.column_stack([B[:,[0,1+j,1+(j+2)%4,5+j,9+j]]@c[j] for j in range(4)])\\n        def resid(a): return (calc(u,a,coeff(u,a,v))-v).ravel()\\n        for a0 in ([2.4,2.4],[1.,1.],[3.2,2.4]):\\n            a=least_squares(resid,a0,bounds=(.2,4.5)).x\\n            err=np.linalg.norm(resid(a)); c=coeff(u,a,v)\\n            if best is None or err < best[0]: best=(err,typ,a,c)\\n    err,typ,a,c=best\\n    def g(t):\\n        if typ==\'sin\': return np.sin(t)\\n        if typ==\'tanh\': return np.tanh(t)\\n        return t/(1+t*t)\\n    def dg(t):\\n        if typ==\'sin\': return np.cos(t)\\n        if typ==\'tanh\': return 1-np.tanh(t)**2\\n        return (1-t*t)/(1+t*t)**2\\n    z=(x-c[:,0])/c[:,1]; j=np.arange(4)\\n    for _ in range(30):\\n        w=np.roll(z,2,axis=1); r=c[:,0]+c[:,1]*z+c[:,2]*w+c[:,3]*g(a[0]*z)+c[:,4]*g(a[1]*w)-x\\n        if np.max(abs(r))<1e-12: break\\n        J=np.zeros((len(z),4,4)); J[:,j,j]=c[:,1]+c[:,3]*a[0]*dg(a[0]*z); J[:,j,(j+2)%4]=c[:,2]+c[:,4]*a[1]*dg(a[1]*w)\\n        z-=np.clip(np.linalg.solve(J,r[...,None])[...,0],-.4,.4)\\n    return z\\n\\n\\ndef _pair(w):\\n    return w[:,[0,0,1,2]]*w[:,[1,3,2,3]]\\n\\n\\ndef _branch(u,v,z,k):\\n    best=None\\n    for typ in (\'sin\',\'tanh\',\'rat\',\'soft\',\'cos\'):\\n        def g(t):\\n            if typ==\'sin\': return np.sin(t)\\n            if typ==\'tanh\': return np.tanh(t)\\n            if typ==\'rat\': return t/(1+t*t)\\n            if typ==\'cos\': return np.cos(t)\\n            return t/(1+abs(t))\\n        for structure in (0,1,2):\\n            def design(w,p):\\n                h=g(p[0]*w+(p[1:] if len(p)>1 else 0)); b=_pair(w) if k==2 else np.empty((len(w),0))\\n                if structure==0:return _ones(np.c_[b,w,h])\\n                if structure==1:return _ones(np.c_[b,w,h,w/(1+.85*abs(w))])\\n                return _ones(np.c_[b,h])\\n            for shift in (False,True):\\n                for start in (.8,1.5,2.4,3.2):\\n                    def res(p):\\n                        A=design(u,p); return (A@_ls(A,v)-v).ravel()\\n                    p=least_squares(res,[start]+([0]*4 if shift else []),bounds=([.09]+([-.65]*4 if shift else []),[4.5]+([.65]*4 if shift else [])),max_nfev=90).x\\n                    A=design(u,p); P=np.linalg.pinv(A); E=A@(P@v)-v; H=(A*P.T).sum(1)\\n                    score=np.mean((E/np.maximum(1-H[:,None],1e-5))**2)**.5*(1.15 if shift else 1)\\n                    if best is None or score<best[0]: best=(score,design(z,p)@(P@v),typ,structure,p)\\n    return best\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    s=inputs[\'__input__\']; z=_align(s[\'panel_a\'],np.vstack((s[\'train\'][\'x\'],s[\'test\'][\'x\'])))\\n    b,c=s[\'panel_b\'],s[\'panel_c\']; l=_ones(z)@_ls(_ones(b[\'u\']),b[\'v\'])\\n    q,r=(_branch(c[\'u\'],c[\'v\'+str(k)],z,k)[1] for k in (1,2))\\n    return dict(z=z,L=l,Q=q,R=r,product=r,cosine=q)\\n", "node_01": "import numpy as np\\nfrom scipy.linalg import cho_factor, cho_solve\\n\\n\\ndef _ls(x, y):\\n    return np.linalg.lstsq(x, y, rcond=None)[0]\\n\\n\\ndef _ones(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _dist(a):\\n    return np.maximum(0, ((a[:, None, :] - a[None, :, :]) ** 2).sum(-1))\\n\\n\\ndef _models(z, l, q, r, p, c):\\n    cross = (p[:, :, None] * c[:, None, :]).reshape(len(z), -1)\\n    ds = _dist(z)\\n    fs = [_ones(np.c_[p, c, cross]), _ones(np.c_[z, p, c, cross]),\\n          _ones(np.c_[p, c, (q * r).sum(1)]), np.ones((len(z), 1))]\\n    for ix, F in enumerate(fs):\\n        lin = F @ F.T / F.shape[1]\\n        for D in [ds, _dist(np.c_[q, r]), ds + .5 * _dist(np.c_[q, r]), ds + .5 * _dist(l)]:\\n            for h in [.9, 1.7, 3., 5.]:\\n                for K in [np.exp(-D / h**2), (1 + D / h**2)**(-2)]:\\n                    for w in [.1, 1., 10., 100.] if ix < 3 else [1.]:\\n                        yield K + w * lin\\n    yield fs[0] @ fs[0].T / fs[0].shape[1]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    t = inputs[\'__input__\'][\'train\']\\n    features = inputs[\'node_00\']\\n    z, l, q, r, product, cosine = (features[k] for k in (\'z\', \'L\', \'Q\', \'R\', \'product\', \'cosine\'))\\n    n = len(t[\'x\'])\\n    traj = t[\'traj\']\\n    _, _, V = np.linalg.svd(traj, full_matrices=False)\\n    H = V[:3]\\n    target = np.c_[t[\'y\'], traj @ H.T]\\n    count = target.shape[1]\\n    models = [[] for _ in range(count)]\\n    I = np.eye(n)\\n    for K in _models(z, l, q, r, product, cosine):\\n        for lam in [1e-6, 1e-4, .002, .025]:\\n            f = cho_factor(K[:n, :n] + lam * I, check_finite=False)\\n            alpha = cho_solve(f, target, check_finite=False)\\n            inv = cho_solve(f, I, check_finite=False).diagonal()\\n            e = np.mean(abs(alpha / inv[:, None]), axis=0)\\n            pred = K[n:, :n] @ alpha\\n            for j in range(count):\\n                models[j].append((e[j], pred[:, j]))\\n    out = []\\n    for m in models:\\n        best = sorted(m, key=lambda v: v[0])[:10]\\n        w = np.exp(-15 * (np.array([a for a, b in best]) / best[0][0] - 1))\\n        out.append(np.average(np.stack([b for a, b in best]), axis=0, weights=w))\\n    pred = np.array(out).T\\n    bilinear = (product[:, :, None] * cosine[:, None, :]).reshape(len(product), -1)\\n    F = _ones(np.c_[product, cosine, bilinear])\\n    Y = np.c_[t[\'y\'], traj]\\n    E = Y - F[:n] @ _ls(F[:n], Y)\\n    _, sig, unit = np.linalg.svd(E, full_matrices=False)\\n    load = unit[0]\\n    if abs(load[0]) > .15 and sig[0] > 5 * sig[1]:\\n        ratio = load[1:] / load[0]\\n        orthogonal = traj - t[\'y\'][:, None] * ratio\\n        state = F[n:] @ _ls(F[:n], orthogonal) + pred[:, 0, None] * ratio\\n    else:\\n        state = pred[:, 1:] @ H\\n    y = pred[:, 0]\\n    end = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state, \'diagnostics_vec\': np.array([y.mean(), y.std(), end.mean(), end.std()])}\\n"}'

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
