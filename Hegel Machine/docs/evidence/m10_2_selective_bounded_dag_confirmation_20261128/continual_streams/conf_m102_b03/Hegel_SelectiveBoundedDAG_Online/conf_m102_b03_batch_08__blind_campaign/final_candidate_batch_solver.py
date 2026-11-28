"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_05", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "1807e56d343167215abbda6a0b05f2984d693070a2481d733ec5758c1ff79426", "node_01": "26f096a670caeff32f7db973a43b8ab7d00bd1c72846d5129f6dd0fa0ebb3ee9"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _ls(x, y):\\n    return np.linalg.lstsq(x, y, rcond=None)[0]\\n\\n\\ndef _ones(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _product_design(w):\\n    return _ones(w[:, [0, 0, 1, 2]] * w[:, [1, 3, 2, 3]])\\n\\n\\ndef _cosine_design(w):\\n    return _ones(np.cos(2.4 * w))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    s = inputs[\'__input__\']\\n    x = np.vstack([s[\'train\'][\'x\'], s[\'test\'][\'x\']])\\n    u, v = (s[\'panel_a\'][k] for k in (\'u\', \'v\'))\\n    i, j = np.triu_indices(len(u), 1)\\n    pars = []\\n    for k in range(u.shape[1]):\\n        d = u[i, k] - u[j, k]\\n        ok = np.abs(d) > 1e-8\\n        m = np.median((v[i[ok], k] - v[j[ok], k]) / d[ok])\\n        r = v[:, k] - m * u[:, k]\\n        r -= np.median(r)\\n        g = np.abs(r) < max(.001, 6 * np.median(abs(r)))\\n        pars.append(_ls(_ones(u[g, k:k+1]), v[g, k]))\\n    off, scale = np.array(pars).T\\n    z = (x - off) / scale\\n    b, c = s[\'panel_b\'], s[\'panel_c\']\\n    product = _product_design(z)\\n    cosine = _cosine_design(z)\\n    L = _ones(z) @ _ls(_ones(b[\'u\']), b[\'v\'])\\n    Q = product @ _ls(_product_design(c[\'u\']), c[\'v1\'])\\n    R = cosine @ _ls(_cosine_design(c[\'u\']), c[\'v2\'])\\n    return {\'z\': z, \'L\': L, \'Q\': Q, \'R\': R, \'product\': product[:, 1:], \'cosine\': cosine[:, 1:]}\\n", "node_01": "import numpy as np\\nfrom scipy.linalg import cho_factor, cho_solve\\n\\n\\ndef _solve(s, f, report=False):\\n    y = np.asarray(s[\'train\'][\'y\']); traj = np.asarray(s[\'train\'][\'traj\']); n = len(y)\\n    z, L, Q, R, p, c = (f[k] for k in (\'z\',\'L\',\'Q\',\'R\',\'product\',\'cosine\'))\\n    m = len(z)\\n    dot = (Q*R).sum(1)\\n    cross = (p[:,:,None]*c[:,None,:]).reshape(m,-1)\\n    B = np.c_[np.ones(m),p,c,cross]\\n    Y = np.c_[y,traj]; beta = np.linalg.lstsq(B[:n],Y,rcond=None)[0]; trend = B@beta\\n    E = Y-trend[:n]; load = np.linalg.svd(E,full_matrices=False)[2][0]; load /= load[0] if abs(load[0])>.1 else 1\\n    target = y+(E@load/(load@load)-E[:,0])*.75 if load[0]==1 else y\\n    mean_blocks = [np.c_[p,c,dot],np.c_[z,p,c,dot],np.c_[p,c,cross],np.c_[z,p,c,cross]]\\n    Gs = [np.ones((m,m))]+[1+A@A.T/A.shape[1] for A in mean_blocks]\\n    def dist(A):\\n        return ((A[:,None,:]-A[None,:,:])**2).sum(2)\\n    dz,dl,dqr=dist(z),dist(L),dist(np.c_[Q,R]); eye=np.eye(n)\\n    vals=[]\\n    def add(K,key,reg):\\n        for lam in reg:\\n            fac=cho_factor(K[:n,:n]+lam*eye,check_finite=False)\\n            a=cho_solve(fac,target,check_finite=False)\\n            d=cho_solve(fac,eye,check_finite=False).diagonal()\\n            e=a/d\\n            score=np.mean(abs(e))+.12*np.sqrt(np.mean(e**2))\\n            vals.append((score,K[n:,:n]@a,key+(lam,)))\\n    scales=[.001,.1,1.,10.,100.]; reg=[1e-8,3e-7,1e-5,3e-4,.003]\\n    for j,D in enumerate([dz,dl,dz+.5*dqr,dqr,dz+.5*dl]):\\n        for h in [1.15,1.7,2.25,3.,4.,5.5,8.,12.]:\\n            for k,K in enumerate([np.exp(-D/h**2),(1+D/h**2)**-2]):\\n                for g,G in enumerate(Gs):\\n                    for w in scales if g else [1.]:\\n                        add(K+w*G,(\'radial\',j,h,k,g,w),reg)\\n    for j,X in enumerate([z,L,np.c_[z,L]]):\\n        for gamma in [.25,.6,1.2]:\\n            for degree in [3,4,5,6,8]:\\n                K=(1+gamma*(X@X.T)/X.shape[1])**degree\\n                for g,G in enumerate(Gs[1:3],1):\\n                    for w in [.1,10.]:\\n                        add(K+w*G,(\'polynomial\',j,gamma,degree,g,w),reg)\\n    for j,X in enumerate([z,L,np.c_[z,L]]):\\n        sq=(X[:,None,:]-X[None,:,:])**2\\n        for h in [.65,1.,1.5,2.4,3.8]:\\n            H=np.exp(-sq/h**2)\\n            A=H.mean(2); P=A*A; whole=np.exp(-sq.sum(2)/(2*h)**2)\\n            for k,K in enumerate([A+.05*whole,A+.15*P+.02*whole,A+.7*P+.08*whole,P+.15*whole]):\\n                for g,G in enumerate(Gs[1:3],1):\\n                    for w in [.1,1.,10.]:\\n                        add(K+w*G,(\'additive\',j,h,k,g,w),reg)\\n    vals.sort(key=lambda v:v[0]); best=vals[:30]; score=np.array([v[0] for v in best]); weights=np.exp(-12*(score/score[0]-1)); pred=np.average(np.array([v[1] for v in best]),axis=0,weights=weights)\\n    state=trend[n:,1:]+(pred-trend[n:,0])[:,None]*load[1:]\\n    if report:\\n        return pred,state,vals,trend,load\\n    return pred,state\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    s, f = inputs[\'__input__\'], inputs[\'node_00\']\\n    y, state = _solve(s,f)\\n    last=state[:,-1]\\n    return {\'y_hat\':y,\'state_pred\':state,\'diagnostics_vec\':np.array([y.mean(),y.std(),last.mean(),last.std()])}\\n"}'

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
