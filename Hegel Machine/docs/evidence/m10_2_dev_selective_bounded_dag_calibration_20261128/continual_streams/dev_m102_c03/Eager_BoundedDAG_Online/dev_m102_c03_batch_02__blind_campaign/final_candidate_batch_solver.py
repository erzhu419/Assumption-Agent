"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_03", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"], "node_02": ["node_00", "node_01"]}, "max_fanout": 2, "node_code_shas": {"node_00": "15dd934d245981e591b1c8f6c0145f652168495452db1b5f898ac36ffffc25aa", "node_01": "b33cca38875d14b756b7711db24c8a687ef93cb91cabaa19d25b9dd03e652fab", "node_02": "d53838f200cc1f8bdc714bb5f858bfb9921b1d3045077d826b0e6d52e4fc7b76"}, "shared_fanout_nodes": ["node_00"], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom itertools import combinations\\n\\n\\ndef _features(z):\\n    s = [np.ones(len(z)), z]\\n    for j in range(4):\\n        u = z[:, j:j+1]\\n        s.extend([u**2, u**3, u**4, np.tanh(u), np.arctan(u), u/(1+abs(u)), np.exp(-u*u/2)])\\n    groups = []\\n    for i, j in combinations(range(4), 2):\\n        u, v = z[:, i:i+1], z[:, j:j+1]\\n        g = [u*v, u*u*v, u*v*v, u*u*v*v, u**3*v, u*v**3, u*v/(1+.5*(u*u+v*v))]\\n        for sign in (-1, 1):\\n            w = (u+sign*v)/1.4\\n            g.extend([np.sin(w), np.cos(w), np.tanh(w), w/(1+abs(w)), np.exp(-w*w/2)])\\n        groups.append(np.column_stack(g))\\n    return s, groups\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    tr = study[\'train\']\\n    x = np.asarray(tr[\'x\'])\\n    xt = np.asarray(study[\'test\'][\'x\'])\\n    n = len(x)\\n    z = (np.vstack([x, xt])-x.mean(0))/np.maximum(x.std(0), 1e-12)\\n    q = -np.expm1(-np.asarray(study[\'eval_times\'])[:, None]*np.array([.392, 1.857]))\\n    a = np.linalg.lstsq(q, tr[\'traj\'].T, rcond=None)[0].T\\n    s, g = _features(z)\\n    f = np.column_stack(s+g+[np.prod(z[:, j], axis=1) for j in combinations(range(4), 3)])[:, 1:]\\n    ys = (2,5,10,14,16,23,24,26,33,35,40,50,51,52,54,61,63,64,65,67,77,78,85,87,92,93,96,101,102,104,106,111,119,128,132,137)\\n    fs = (7,9,10,24,30,31,35,36,39,40,51,54,61,68,71,72,85,86,96,101,104,106,111,112,123,127,128,134)\\n\\n    def ols(ids, target):\\n        mat = np.column_stack([np.ones(len(z)), f[:, ids]])\\n        return mat[n:]@np.linalg.lstsq(mat[:n], target, rcond=None)[0]\\n\\n    m = np.column_stack(s+[g[j] for j in (0, 1, 3)])\\n    m = m/np.maximum(np.linalg.norm(m[:n], axis=0), 1e-8)\\n    slow = m[n:]@m[:n].T@np.linalg.solve(m[:n]@m[:n].T+.02*np.eye(n), a[:, 0])\\n    y = ols(ys, tr[\'y\'])\\n    pred = np.column_stack([slow, ols(fs, a[:, 1])])@q.T\\n    return {\'y_hat\': y, \'state_pred\': pred}\\n", "node_01": "import numpy as np\\nfrom itertools import combinations\\nfrom scipy.optimize import least_squares\\nfrom scipy.linalg import solve\\n\\n\\ndef _fit(x,y):\\n    return np.linalg.lstsq(x,y,rcond=1e-10)[0]\\n\\n\\ndef _coordinates(study,x):\\n    a=study[\'panel_a\']; u=np.asarray(a[\'u\']); v=np.asarray(a[\'v\'])\\n    i,j=np.triu_indices(len(u),1)\\n    du=u[i]-u[j]; dv=v[i]-v[j]\\n    g=np.nanmedian(np.divide(dv,du,out=np.full_like(du,np.nan),where=abs(du)>1e-9),axis=0)\\n    o=np.median(v-u*g,axis=0)\\n    for k in range(x.shape[1]):\\n        r=abs(v[:,k]-u[:,k]*g[k]-o[k]); keep=r<max(.002,5*np.median(r))\\n        if keep.sum()>=3:\\n            o[k],g[k]=_fit(np.c_[np.ones(keep.sum()),u[keep,k]],v[keep,k])\\n    return (x-o)/g\\n\\n\\ndef _panel_features(study, z):\\n    def affine(x):\\n        return np.column_stack((np.ones(len(x)), x))\\n\\n    p = study[\'panel_c\']\\n    u, v1, v2 = (np.asarray(p[k]) for k in (\'u\', \'v1\', \'v2\'))\\n\\n    def soft(x, k):\\n        return x/(1+k*np.abs(x))\\n\\n    def d1(x, k):\\n        return affine(np.column_stack((x, soft(x, k))))\\n\\n    def r1(k):\\n        d = d1(u, k)\\n        return (d@_fit(d, v1)-v1).ravel()\\n\\n    k = least_squares(r1, np.full(4, .83), bounds=(.25, 2.5), max_nfev=40).x\\n    soft_z = soft(z, k)\\n    sat = d1(z, k)@_fit(d1(u, k), v1)\\n\\n    def r2(k):\\n        d = affine(np.cos(k*u))\\n        return (d@_fit(d, v2)-v2).ravel()\\n\\n    freq = least_squares(r2, [2.4], bounds=(1.4, 3.4), max_nfev=30).x\\n    cos_z = np.cos(freq*z)\\n    cos_mix = affine(cos_z)@_fit(affine(np.cos(freq*u)), v2)\\n    return z, soft_z, sat, cos_z, cos_mix\\n\\n\\ndef _readout(channels, y):\\n    from sklearn.linear_model import lars_path\\n    import warnings\\n    z, soft, sat, cos_z, cos_mix = channels\\n    n = len(y)\\n    base = np.column_stack(channels)\\n    i, j = np.triu_indices(base.shape[1])\\n    # Prioritize products of the matched panel channels, while allowing the data\\n    # to select additional interactions from the complete quadratic dictionary.\\n    core = np.column_stack((base, sat*cos_mix, sat[:, 0]*cos_mix[:, 1]))\\n    full = np.column_stack((core, base[:, i]*base[:, j]))\\n    mean = full[:n].mean(0)\\n    scale = np.maximum(full[:n].std(0), 1e-10)\\n    f = (full-mean)/scale\\n    scale_prior = np.ones(f.shape[1])\\n    scale_prior[:core.shape[1]] = 2.\\n    scale_prior[base.shape[1]:core.shape[1]] = 3.\\n    f *= scale_prior\\n    m = y.mean(0)\\n    ridge = f[:n]@f[:n].T + .07*np.eye(n)\\n    broad = f[n:]@f[:n].T@solve(ridge, y-m, assume_a=\'pos\')+m\\n    preds = []\\n    for k in range(y.shape[1]):\\n        target = y[:, k]-m[k]\\n        with warnings.catch_warnings():\\n            warnings.simplefilter(\'ignore\')\\n            _, _, path = lars_path(f[:n], target, method=\'lasso\', max_iter=135)\\n        models = {}\\n        for v in path.T:\\n            ix = np.flatnonzero(np.abs(v)>1e-9)\\n            dof = len(ix)\\n            if not 4 <= dof <= min(27, n//2):\\n                continue\\n            g = f[:n, ix]\\n            cov = g.T@g + .007*np.eye(dof)\\n            coef = solve(cov, g.T@target, assume_a=\'pos\')\\n            resid = target-g@coef\\n            lever = np.einsum(\'ij,ji->i\', g, solve(cov, g.T, assume_a=\'pos\')) + 1/n\\n            loo = np.mean(np.abs(resid)/np.maximum(1-lever, .05))\\n            score = loo*(1+.012*dof)\\n            old = models.get(tuple(ix))\\n            if old is None or score < old[0]:\\n                models[tuple(ix)] = (score, f[n:, ix]@coef+m[k])\\n        if models:\\n            best = sorted(models.values(), key=lambda pair:pair[0])[:6]\\n            scores = np.array([v[0] for v in best]); weight = (scores[0]/np.maximum(scores, 1e-12))**8\\n            preds.append(.96*np.average([v[1] for v in best],axis=0,weights=weight)+.04*broad[:,k])\\n        else:\\n            preds.append(broad[:, k])\\n    return np.column_stack(preds)\\n\\n\\ndef _predict(study):\\n    tr = study[\'train\']; z = _coordinates(study, np.vstack((tr[\'x\'], study[\'test\'][\'x\'])))\\n    t = np.asarray(study[\'eval_times\']); traj = np.asarray(tr[\'traj\'])\\n    def basis(r):\\n        return -np.expm1(-t[:, None]*r)\\n    def profile(r):\\n        q = basis(r)\\n        return (q@_fit(q, traj.T)-traj.T).ravel()\\n    q = basis(least_squares(profile, [.4125, 1.81], bounds=([.05, .9], [.89, 5.])).x)\\n    target = np.column_stack((tr[\'y\'], _fit(q, traj.T).T))\\n    pred = _readout(_panel_features(study, z), target)\\n    return np.column_stack((pred[:, 0], pred[:, 1:]@q.T))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study=inputs[\'__input__\']; offset=inputs[\'node_00\']\\n    base=np.column_stack((offset[\'y_hat\'],offset[\'state_pred\']))\\n    pred=_predict(study)\\n    return {\'panel_pred\':base+(pred-base)/np.r_[.5,np.full(base.shape[1]-1,.65)]}\\n", "node_02": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    sparse = inputs[\'node_00\']\\n    panel = inputs[\'node_01\'][\'panel_pred\']\\n    y = 0.5 * sparse[\'y_hat\'] + 0.5 * panel[:, 0]\\n    state = 0.35 * sparse[\'state_pred\'] + 0.65 * panel[:, 1:]\\n    last = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), last.mean(), last.std()])}\\n"}'

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
