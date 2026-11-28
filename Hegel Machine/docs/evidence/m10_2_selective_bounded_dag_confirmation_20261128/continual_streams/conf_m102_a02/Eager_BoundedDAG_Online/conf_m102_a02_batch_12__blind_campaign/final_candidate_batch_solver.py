"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_14", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"], "node_02": ["node_00", "node_01"]}, "max_fanout": 2, "node_code_shas": {"node_00": "f253fc254797595e6661454a2639d9661e7e441e165ab278d8a837bcdabb9345", "node_01": "cd82a2d57b05f67d9822073f47ed2b985ce4459584912d225f07001c207b31a8", "node_02": "c511ca91569f4f92442799147e67ed8f097fbd7e5643c07b827afca8c89039dd"}, "shared_fanout_nodes": ["node_00"], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "from itertools import combinations_with_replacement\\nimport numpy as np\\n\\n\\ndef _design(x):\\n    x = np.asarray(x, dtype=float)\\n    affine = np.column_stack((np.ones(len(x)), x))\\n    quadratic = [x[:, i] * x[:, j]\\n                 for i, j in combinations_with_replacement(range(x.shape[1]), 2)]\\n    return affine, np.column_stack((affine, *quadratic))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    a, q = _design(study[\'train\'][\'x\'])\\n    a_test, q_test = _design(study[\'test\'][\'x\'])\\n    return {\'a\': a, \'q\': q, \'a_test\': a_test, \'q_test\': q_test}\\n", "node_01": "\\"\\"\\"Fit paired readouts in robustly calibrated panel coordinates.\\"\\"\\"\\nfrom itertools import combinations_with_replacement\\nimport numpy as np\\nfrom scipy.optimize import minimize_scalar\\n\\n\\ndef _fit(a, b):\\n    return np.linalg.lstsq(a, b, rcond=1e-10)[0]\\n\\n\\ndef _calibrate(panel, x):\\n    u, v = (np.asarray(panel[k], float) for k in (\'u\', \'v\'))\\n    i, j = np.triu_indices(len(u), 1)\\n    scale = np.empty(u.shape[1])\\n    offset = np.empty_like(scale)\\n    for k, (a, b) in enumerate(zip(u.T, v.T)):\\n        du = a[j] - a[i]\\n        ok = np.abs(du) > 1e-8\\n        m = np.median((b[j[ok]] - b[i[ok]]) / du[ok])\\n        r = b - m*a\\n        r -= np.median(r)\\n        keep = np.abs(r) <= max(9*np.median(np.abs(r)), 1e-6)\\n        offset[k], scale[k] = _fit(np.c_[np.ones(keep.sum()), a[keep]], b[keep])\\n    return (x-offset)/scale\\n\\n\\ndef _panel_map(panel, key, f, bounds, linear):\\n    u, v = (np.asarray(panel[k], float) for k in (\'u\', key))\\n    def design(x, w):\\n        if linear:\\n            return np.c_[np.ones(len(x)), x, f(w*x)]\\n        return np.c_[np.ones(len(x)), f(w*x)]\\n    def loss(w):\\n        a = design(u, w)\\n        return np.mean((a @ _fit(a,v) - v)**2)\\n    grid = np.linspace(*bounds, 81)\\n    k = np.argmin([loss(w) for w in grid])\\n    w = minimize_scalar(loss, bounds=(grid[max(k-1,0)],grid[min(k+1,len(grid)-1)]), method=\'bounded\', options={\'xatol\':1e-10}).x\\n    c = _fit(design(u,w),v)\\n    return lambda x: design(x,w) @ c\\n\\n\\ndef _features(study, x):\\n    z = _calibrate(study[\'panel_a\'], x)\\n    g = _panel_map(study[\'panel_b\'], \'v\', np.cos, (0.4,5.0), False)(z)\\n    p = _panel_map(study[\'panel_c\'], \'v1\', np.tanh, (0.4,4.0), True)(g)\\n    q = _panel_map(study[\'panel_c\'], \'v2\', lambda a: a*a/(1+a*a), (0.15,1.5), True)(g)\\n    quadratic = np.column_stack([g[:,i]*g[:,j] for i,j in combinations_with_replacement(range(g.shape[1]),2)])\\n    return np.c_[np.ones(len(g)), g, quadratic, p, q]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study, raw = inputs[\'__input__\'], inputs[\'node_00\']\\n    train = study[\'train\']\\n    a, b = raw[\'a\'], raw[\'a_test\']\\n    m = _features(study, np.concatenate((a[:,1:], b[:,1:]),axis=0))\\n    y, t = np.asarray(train[\'y\']), np.asarray(train[\'traj\'])\\n    modes = np.linalg.svd(t, full_matrices=False)[2][:2]\\n    targets = np.c_[y, t @ modes.T]\\n    c = _fit(m[:len(a)], targets)\\n    return {\'test_features\': m[len(a):], \'coefficients\': np.c_[c[:,0],c[:,1:] @ modes]}\\n", "node_02": "\\"\\"\\"Apply the fitted readouts to the calibrated test design.\\"\\"\\"\\nimport numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    raw, fitted = inputs[\'node_00\'], inputs[\'node_01\']\\n    z = fitted[\'test_features\']\\n    if len(z) != len(raw[\'a_test\']):\\n        raise ValueError(\'Calibrated design must retain all test rows\')\\n    values = z @ fitted[\'coefficients\']\\n    endpoint, state = values[:, 0], values[:, 1:]\\n    last = state[:, -1]\\n    return {\'y_hat\': endpoint, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([endpoint.mean(), endpoint.std(), last.mean(), last.std()])}\\n"}'

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
