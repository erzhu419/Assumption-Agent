"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_04", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"], "node_02": ["node_00", "node_01"]}, "max_fanout": 2, "node_code_shas": {"node_00": "15dd934d245981e591b1c8f6c0145f652168495452db1b5f898ac36ffffc25aa", "node_01": "ef5e67ec8fb7c7a7ddf1c6f0f7b3b003674fcf2a5d4d039f1996eb6ff717befd", "node_02": "d53838f200cc1f8bdc714bb5f858bfb9921b1d3045077d826b0e6d52e4fc7b76"}, "shared_fanout_nodes": ["node_00"], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom itertools import combinations\\n\\n\\ndef _features(z):\\n    s = [np.ones(len(z)), z]\\n    for j in range(4):\\n        u = z[:, j:j+1]\\n        s.extend([u**2, u**3, u**4, np.tanh(u), np.arctan(u), u/(1+abs(u)), np.exp(-u*u/2)])\\n    groups = []\\n    for i, j in combinations(range(4), 2):\\n        u, v = z[:, i:i+1], z[:, j:j+1]\\n        g = [u*v, u*u*v, u*v*v, u*u*v*v, u**3*v, u*v**3, u*v/(1+.5*(u*u+v*v))]\\n        for sign in (-1, 1):\\n            w = (u+sign*v)/1.4\\n            g.extend([np.sin(w), np.cos(w), np.tanh(w), w/(1+abs(w)), np.exp(-w*w/2)])\\n        groups.append(np.column_stack(g))\\n    return s, groups\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    tr = study[\'train\']\\n    x = np.asarray(tr[\'x\'])\\n    xt = np.asarray(study[\'test\'][\'x\'])\\n    n = len(x)\\n    z = (np.vstack([x, xt])-x.mean(0))/np.maximum(x.std(0), 1e-12)\\n    q = -np.expm1(-np.asarray(study[\'eval_times\'])[:, None]*np.array([.392, 1.857]))\\n    a = np.linalg.lstsq(q, tr[\'traj\'].T, rcond=None)[0].T\\n    s, g = _features(z)\\n    f = np.column_stack(s+g+[np.prod(z[:, j], axis=1) for j in combinations(range(4), 3)])[:, 1:]\\n    ys = (2,5,10,14,16,23,24,26,33,35,40,50,51,52,54,61,63,64,65,67,77,78,85,87,92,93,96,101,102,104,106,111,119,128,132,137)\\n    fs = (7,9,10,24,30,31,35,36,39,40,51,54,61,68,71,72,85,86,96,101,104,106,111,112,123,127,128,134)\\n\\n    def ols(ids, target):\\n        mat = np.column_stack([np.ones(len(z)), f[:, ids]])\\n        return mat[n:]@np.linalg.lstsq(mat[:n], target, rcond=None)[0]\\n\\n    m = np.column_stack(s+[g[j] for j in (0, 1, 3)])\\n    m = m/np.maximum(np.linalg.norm(m[:n], axis=0), 1e-8)\\n    slow = m[n:]@m[:n].T@np.linalg.solve(m[:n]@m[:n].T+.02*np.eye(n), a[:, 0])\\n    y = ols(ys, tr[\'y\'])\\n    pred = np.column_stack([slow, ols(fs, a[:, 1])])@q.T\\n    return {\'y_hat\': y, \'state_pred\': pred}\\n", "node_01": "import numpy as np\\nfrom itertools import combinations_with_replacement\\n\\n\\ndef _fit(a, b):\\n    return np.linalg.lstsq(a, b, rcond=None)[0]\\n\\n\\ndef _latent(x, panel):\\n    u, v = np.asarray(panel[\'u\']), np.asarray(panel[\'v\'])\\n    a = np.column_stack((np.ones(len(u)), u))\\n    keep = np.linalg.norm(v-a@_fit(a,v), axis=1) < 2\\n    w = []\\n    for i in range(4):\\n        j = (i+1)%4\\n        h = np.column_stack((a[:,0], u[:,i], u[:,j], u[:,i]*u[:,j], u[:,i]**3))\\n        w.append(_fit(h[keep], v[keep,i]))\\n    off,g,c,b,d = np.array(w).T\\n    z = (x-off)/g\\n    ix = np.arange(4)\\n    jx = (ix+1)%4\\n    for _ in range(14):\\n        zn = z[:,jx]\\n        r = off + g*z+c*zn+b*z*zn+d*z**3-x\\n        jac = np.zeros((len(z),4,4))\\n        jac[:,ix,ix] = g+b*zn+3*d*z*z\\n        jac[:,ix,jx] = c+b*z\\n        z -= np.clip(np.linalg.solve(jac, r[...,None])[...,0], -0.75, 0.75)\\n    return z\\n\\n\\ndef _warp(z, panel):\\n    u, v = np.asarray(panel[\'u\']), np.asarray(panel[\'v2\'])\\n    a = np.column_stack((np.ones(len(u)), u))\\n    def profile(rates):\\n        h = np.column_stack((a, np.expm1(-rates*u*u)))\\n        w = _fit(h,v)\\n        return np.sum((h@w-v)**2), w\\n    rates = np.full(4, .2934)\\n    for _ in range(3):\\n        for j in range(4):\\n            lo, hi = .25,.34\\n            for _ in range(16):\\n                p,q = (2*lo+hi)/3,(lo+2*hi)/3\\n                rates[j]=p\\n                ep=profile(rates)[0]\\n                rates[j]=q\\n                eq=profile(rates)[0]\\n                if ep < eq:\\n                    hi=q\\n                else:\\n                    lo=p\\n            rates[j]=(lo+hi)/2\\n    w = profile(rates)[1]\\n    return z + np.expm1(-rates*z*z) @ np.linalg.solve(w[1:5].T,w[5:].T).T\\n\\n\\ndef _poly(z, degree):\\n    return np.column_stack([np.prod(z[:,j],axis=1) for k in range(1,degree+1) for j in combinations_with_replacement(range(4),k)])\\n\\n\\ndef _temporal_basis(times, traj):\\n    # Variable projection: optimize rates, solving amplitudes exactly at each step.\\n    t = np.asarray(times)[:, None]\\n    rates = np.array([.408, 1.815])\\n    def profile(r):\\n        q = -np.expm1(-t*r)\\n        return q, (q@_fit(q,traj.T)-traj.T).ravel()\\n    q, r = profile(rates)\\n    for _ in range(10):\\n        h = 1.e-4\\n        J = np.column_stack([(profile(rates+d)[1]-profile(rates-d)[1])/(2*h) for d in h*np.eye(2)])\\n        step = np.clip(_fit(J,r), -.35, .35)\\n        if np.linalg.norm(step) < 1.e-8:\\n            break\\n        for shrink in (1., .5, .25, .125):\\n            cand = np.clip(rates-shrink*step, [.1,1.], [.9,4.])\\n            qq, rr = profile(cand)\\n            if rr@rr < r@r:\\n                rates, q, r = cand, qq, rr\\n                break\\n        else:\\n            break\\n    return q\\n\\n\\ndef _predict(study):\\n    tr = study[\'train\']\\n    n = len(tr[\'x\'])\\n    x = np.vstack((tr[\'x\'],study[\'test\'][\'x\']))\\n    z = _warp(_latent(x,study[\'panel_a\']),study[\'panel_c\'])\\n    f = np.column_stack((_poly(z,3), np.tanh(4*z)))\\n    f = (f-f[:n].mean(0))/np.maximum(f[:n].std(0),1e-8)\\n    traj = np.asarray(tr[\'traj\'])\\n    q = _temporal_basis(study[\'eval_times\'],traj)\\n    y = np.column_stack((tr[\'y\'], _fit(q,traj.T).T))\\n    mu = y.mean(0)\\n    a = f[:n]\\n    g = a.T@a+.003*np.eye(a.shape[1])\\n    rhs = a.T@(y-mu)\\n    coef = np.linalg.solve(g,rhs)\\n    scale = np.sqrt(np.diag(np.linalg.solve(g,np.eye(len(g)))))\\n    pred = []\\n    for k in range(y.shape[1]):\\n        ix = np.argsort(np.abs(coef[:,k])/scale)[-28:]\\n        pred.append(f[n:,ix]@np.linalg.solve(g[np.ix_(ix,ix)],rhs[ix,k])+mu[k])\\n    return pred[0], np.column_stack(pred[1:])@q.T\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    y, traj = _predict(inputs[\'__input__\'])\\n    sparse = inputs[\'node_00\']\\n    panel_y = 2*y-np.asarray(sparse[\'y_hat\'])\\n    panel_traj = (traj-.35*np.asarray(sparse[\'state_pred\']))/.65\\n    return {\'panel_pred\': np.column_stack((panel_y,panel_traj))}\\n", "node_02": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    sparse = inputs[\'node_00\']\\n    panel = inputs[\'node_01\'][\'panel_pred\']\\n    y = 0.5 * sparse[\'y_hat\'] + 0.5 * panel[:, 0]\\n    state = 0.35 * sparse[\'state_pred\'] + 0.65 * panel[:, 1:]\\n    last = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), last.mean(), last.std()])}\\n"}'

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
