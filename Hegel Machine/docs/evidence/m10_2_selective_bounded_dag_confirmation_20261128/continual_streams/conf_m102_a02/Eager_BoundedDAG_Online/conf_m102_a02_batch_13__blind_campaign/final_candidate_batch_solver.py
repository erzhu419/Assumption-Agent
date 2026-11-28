"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_15", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"], "node_02": ["node_00", "node_01"]}, "max_fanout": 2, "node_code_shas": {"node_00": "f253fc254797595e6661454a2639d9661e7e441e165ab278d8a837bcdabb9345", "node_01": "2559b687cfea2bb7e750c2332b12ca5d1d6881a2dbea6eab1e6823102312c107", "node_02": "c511ca91569f4f92442799147e67ed8f097fbd7e5643c07b827afca8c89039dd"}, "shared_fanout_nodes": ["node_00"], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "from itertools import combinations_with_replacement\\nimport numpy as np\\n\\n\\ndef _design(x):\\n    x = np.asarray(x, dtype=float)\\n    affine = np.column_stack((np.ones(len(x)), x))\\n    quadratic = [x[:, i] * x[:, j]\\n                 for i, j in combinations_with_replacement(range(x.shape[1]), 2)]\\n    return affine, np.column_stack((affine, *quadratic))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    a, q = _design(study[\'train\'][\'x\'])\\n    a_test, q_test = _design(study[\'test\'][\'x\'])\\n    return {\'a\': a, \'q\': q, \'a_test\': a_test, \'q_test\': q_test}\\n", "node_01": "\\"\\"\\"Robust inverse calibration and independent scalar and temporal readouts.\\"\\"\\"\\nfrom itertools import combinations_with_replacement\\n\\nimport numpy as np\\nfrom scipy.optimize import least_squares, minimize_scalar\\n\\n\\ndef fit(a, b):\\n    return np.linalg.lstsq(a, b, rcond=1e-9)[0]\\n\\n\\ndef coordinates(study, raw):\\n    x = np.vstack((raw[\'a\'][:, 1:], raw[\'a_test\'][:, 1:]))\\n    p = study[\'panel_a\']\\n    u, v = np.asarray(p[\'u\']), np.asarray(p[\'v\'])\\n    successor = np.roll(u, -1, axis=1)\\n    coefficients = []\\n    for a, b, target in zip(u.T, successor.T, v.T):\\n        d = np.column_stack((np.ones(len(a)), a, b, a*b, a**3))\\n        c = least_squares(lambda c: d@c-target, fit(d, target), loss=\'cauchy\', f_scale=.002).x\\n        m = np.abs(d@c-target) < .002\\n        coefficients.append(fit(d[m], target[m]) if m.sum() >= d.shape[1] else c)\\n    c = np.asarray(coefficients).T\\n    def forward(z):\\n        w = np.roll(z, -1)\\n        return c[0] + c[1]*z + c[2]*w + c[3]*z*w + c[4]*z**3\\n    def jac(z):\\n        j = np.diag(c[1] + c[3]*np.roll(z, -1) + 3*c[4]*z*z)\\n        i = np.arange(len(z))\\n        j[i, (i+1)%len(z)] = c[2]+c[3]*z\\n        return j\\n    z = np.array([least_squares(lambda z: forward(z)-row, row, jac=jac, gtol=1e-11).x for row in x])\\n    p = study[\'panel_b\']\\n    def design(a, w):\\n        return np.column_stack((np.ones(len(a)), np.cos(w*a)))\\n    def loss(w):\\n        a = design(p[\'u\'], w)\\n        return np.mean((a@fit(a, p[\'v\'])-p[\'v\'])**2)\\n    w = minimize_scalar(loss, bounds=(2.,3.), method=\'bounded\').x\\n    b = design(z, w)\\n    return b[:,1:], b@fit(design(p[\'u\'],w),p[\'v\'])\\n\\n\\ndef endpoint(u, g, y, spec):\\n    base, numerator, penalty, freq, shift, kernel = spec\\n    n = len(y)\\n    a = np.column_stack((np.ones(len(g)), g))\\n    cross = g[:,:2]*g[:,2:]\\n    quad = np.column_stack([g[:,i]*g[:,j] for i,j in combinations_with_replacement(range(g.shape[1]),2)])\\n    d = (u[:,0]-shift*u[:,2])**2\\n    def make(w):\\n        r = 1/(1+w*d)\\n        terms = {\'one\': u[:,2:3], \'pair\': u[:,[0,2]], \'all\': u, \'const\': np.column_stack((np.ones(len(u)),u[:,2])), \'cross\': np.column_stack((u[:,2],cross))}[numerator] if numerator != \'none\' else np.empty((len(u),0))\\n        core = np.column_stack((a, cross, r[:,None]*terms))\\n        if base == \'cross\': return core, core.shape[1]\\n        return np.column_stack((core,quad)), core.shape[1]\\n    def train(w, loo=False):\\n        X, k = make(w)\\n        if base == \'cross\' or penalty == 0:\\n            if not loo: return X, fit(X[:n],y)\\n            U,S,_ = np.linalg.svd(X[:n],full_matrices=False)\\n            U = U[:,S>S[0]*1e-9]; h=(U**2).sum(1); e=y-U@(U.T@y)\\n        else:\\n            M = X[:n].T@X[:n] + np.diag(np.r_[np.zeros(k),np.full(X.shape[1]-k,penalty)])\\n            L = np.linalg.pinv(M,rcond=1e-12)@X[:n].T\\n            c=L@y\\n            if not loo: return X,c\\n            h=(X[:n]*L.T).sum(1); e=y-X[:n]@c\\n        return np.mean((e/np.maximum(1-h,.02))**2)\\n    if freq == 0:\\n        ws = [.10,.22,.4,.65,1.,2.,4.]\\n        w0=min(ws,key=lambda w: train(w,True)); j=ws.index(w0)\\n        w = minimize_scalar(lambda w:train(w,True),bounds=(ws[max(0,j-1)],ws[min(len(ws)-1,j+1)]),method=\'bounded\',options={\'xatol\':.001}).x\\n    else:\\n        w=freq\\n    X,c=train(w)\\n    if kernel:\\n        # A small smooth residual correction, projected off the structural trend.\\n        from scipy.spatial.distance import cdist\\n        X, k = make(w)\\n        X = X[:,:k]\\n        U, S, _=np.linalg.svd(X[:n],full_matrices=False)\\n        U=U[:,S>S[0]*1e-9]; J=np.eye(n)-U@U.T\\n        coords=u/np.maximum(u[:n].std(0),1e-8)\\n        dist=cdist(coords[:n],coords,\'sqeuclidean\')\\n        best=(np.inf,None,None)\\n        for width in [.015,.05,.15,.4]:\\n            K=np.exp(-width*dist); A=K[:,:n]; lam,V=np.linalg.eigh(J@A@J); V=J@V; Vy=V.T@y\\n            for reg in [.0001,.003,.03,.15]:\\n                R=V/(np.maximum(lam,0)+reg); alpha=R@Vy\\n                err=np.mean(np.abs(alpha/np.maximum((R*V).sum(1),1e-9)))\\n                if err < best[0]: best=(err,K,alpha)\\n        _,K,alpha=best\\n        return np.column_stack((X,K.T)), np.r_[fit(X[:n],y-K[:,:n]@alpha),alpha]\\n    return X,c\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study, raw = inputs[\'__input__\'], inputs[\'node_00\']\\n    u,g = coordinates(study,raw)\\n    y, traj = study[\'train\'][\'y\'], study[\'train\'][\'traj\']\\n    n=len(y)\\n    X,c=endpoint(u,g,y,SPEC)\\n    a=np.column_stack((np.ones(len(g)),g))\\n    features=np.column_stack((a,X))\\n    coefficients=np.zeros((features.shape[1],1+traj.shape[1]))\\n    coefficients[a.shape[1]:,0]=c\\n    coefficients[:a.shape[1],1:]=fit(a[:n],traj)\\n    return {\'test_features\':features[n:], \'coefficients\':coefficients}\\n\\n\\nSPEC = (\'quad\', \'one\', 3.0, 0, 1.25, False)\\n", "node_02": "\\"\\"\\"Apply the fitted readouts to the calibrated test design.\\"\\"\\"\\nimport numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    raw, fitted = inputs[\'node_00\'], inputs[\'node_01\']\\n    z = fitted[\'test_features\']\\n    if len(z) != len(raw[\'a_test\']):\\n        raise ValueError(\'Calibrated design must retain all test rows\')\\n    values = z @ fitted[\'coefficients\']\\n    endpoint, state = values[:, 0], values[:, 1:]\\n    last = state[:, -1]\\n    return {\'y_hat\': endpoint, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([endpoint.mean(), endpoint.std(), last.mean(), last.std()])}\\n"}'

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
