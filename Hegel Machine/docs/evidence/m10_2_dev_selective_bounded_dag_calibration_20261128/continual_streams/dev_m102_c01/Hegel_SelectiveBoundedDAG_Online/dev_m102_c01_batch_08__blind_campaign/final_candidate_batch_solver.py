"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_01", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "cee793a10b3c7f9ae486629e9254feb633e38e8f01361b060939b5e15af7af57"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Panel-calibrated spatial prediction with a shared trajectory response.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.linalg import solve\\nfrom scipy.optimize import minimize_scalar\\nfrom scipy.spatial.distance import cdist\\n\\n\\ndef _coordinates(study, x):\\n    a = study[\'panel_a\']\\n    u, v = np.asarray(a[\'u\']), np.asarray(a[\'v\'])\\n    i, j = np.triu_indices(len(u), 1)\\n    du = u[i] - u[j]\\n    slope = np.median((v[i] - v[j]) / np.where(abs(du) > 1e-12, du, np.inf), axis=0)\\n    offset = np.median(v - u*slope, axis=0)\\n    for _ in range(3):\\n        r = np.linalg.norm(v - u*slope - offset, axis=1)\\n        keep = r <= max(6*np.median(r), 0.0005)\\n        if keep.sum() < 5:\\n            break\\n        for k in range(u.shape[1]):\\n            slope[k], offset[k] = np.linalg.lstsq(np.c_[u[keep,k], np.ones(keep.sum())], v[keep,k], rcond=None)[0]\\n    b = study[\'panel_b\']\\n    bu, bv = np.asarray(b[\'u\']), np.asarray(b[\'v\'])\\n    bmap = np.linalg.lstsq(np.c_[np.ones(len(bu)), bu], bv, rcond=None)[0][1:]\\n    safe = np.where(abs(slope)>1e-8, slope, 1.)\\n    return {\'x\': x, \'inverse\': (x-offset)/safe, \'forward\': x*slope+offset,\\n            \'b\': x@bmap, \'bi\': ((x-offset)/safe)@bmap, \'bf\': (x*slope+offset)@bmap}\\n\\n\\ndef _kernel(x, width, quadratic=0.):\\n    ip = x @ x.T\\n    return 1. + ip + 2*np.exp(-width*cdist(x,x,\'sqeuclidean\')) + quadratic*(ip*ip)\\n\\n\\ndef _predict(K, n, target, ridge):\\n    dual = solve(K[:n,:n] + ridge*np.eye(n), target, assume_a=\'pos\')\\n    return K[n:,:n] @ dual\\n\\n\\ndef _rational_panel(panel):\\n    u, v = (np.asarray(panel[key], dtype=float) for key in (\'u\', \'v2\'))\\n\\n    def features(x, width):\\n        xx = x*x\\n        return np.c_[np.ones(len(x)), x, xx/(1. + width*xx)]\\n\\n    def loo_error(width):\\n        q, _ = np.linalg.qr(features(u, width), mode=\'reduced\')\\n        residual = v - q @ (q.T @ v)\\n        leverage = np.sum(q*q, axis=1, keepdims=True)\\n        return np.mean((residual/np.maximum(1. - leverage, 1e-8))**2)\\n\\n    width = minimize_scalar(loo_error, bounds=(.02, 2.), method=\'bounded\').x\\n    coef = np.linalg.lstsq(features(u, width), v, rcond=None)[0]\\n    return lambda x: features(x, width) @ coef\\n\\n\\ndef _auxiliary_kernel(op, x, n):\\n    z = op(x)\\n    z -= z[:n].mean(axis=0)\\n    ip = z @ z.T\\n    spread = 2.*np.sum(z[:n]**2)/n\\n    return ip + .2*ip**2 + np.exp(-.72*cdist(z,z,\'sqeuclidean\')/max(spread, 1e-12))\\n\\n\\ndef solve_batch(study: dict) -> dict:\\n    train = study[\'train\']\\n    x = np.vstack([train[\'x\'], study[\'test\'][\'x\']])\\n    n = len(train[\'x\'])\\n    z = _coordinates(study, x)\\n    op = _rational_panel(study[\'panel_c\'])\\n    cf = _auxiliary_kernel(op, z[\'bf\'], n)\\n    ci = _auxiliary_kernel(op, z[\'bi\'], n)\\n    smooth = _kernel(z[\'bf\'], .18)\\n    Y = np.asarray(train[\'y\'])\\n    y = .6*_predict(smooth + .5*cf, n, Y, .001) + .4*_predict(smooth + .1*cf, n, Y, .001)\\n    T = np.asarray(train[\'traj\'])\\n    _, _, vt = np.linalg.svd(T, full_matrices=False)\\n    mode = vt[0]\\n    a = T @ mode\\n    poly = (z[\'forward\'] @ z[\'forward\'].T)**2\\n    k = _kernel(z[\'bf\'], .18, .2) + .3*poly\\n    amp = .5*_predict(k + .5*ci, n, a, .004) + .5*_predict(k + .1*ci, n, a, .004)\\n    state = np.outer(amp, mode) + _predict(_kernel(z[\'x\'], .25), n, T - np.outer(a, mode), .1)\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), state[:,-1].mean(), state[:,-1].std()])}\\n"}'

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
