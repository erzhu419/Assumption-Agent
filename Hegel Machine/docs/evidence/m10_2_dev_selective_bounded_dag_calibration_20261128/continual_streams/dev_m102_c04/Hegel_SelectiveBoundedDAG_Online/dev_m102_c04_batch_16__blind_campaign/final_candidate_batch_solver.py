"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_10", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__"], "node_02": ["node_00", "node_01"]}, "max_fanout": 1, "node_code_shas": {"node_00": "279e716757cdb6f5c85c387c86751c280a4516f4dbee4cb82984220e38140875", "node_01": "1879b26e831195dba0cd078d4cfeb0456354c16168fc63c9ba65b58818a39a82", "node_02": "afce3e7495447f9c2b5d6be728eaf63eaff35414616b8e651f35d40a6ef144c5"}, "shared_fanout_nodes": [], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Recover calibrated spatial coordinates and construct panel-C predictors.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _fit(a, y):\\n    return np.linalg.lstsq(a, y, rcond=1e-10)[0]\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _calibrate(panel, x):\\n    u, v = np.asarray(panel[\'u\']), np.asarray(panel[\'v\'])\\n    distance = np.linalg.norm(u-v, axis=1)\\n    keep = distance < max(4.0, 4*np.median(distance))\\n    u, v = u[keep], v[keep]\\n    partner = np.array([2, 3, 0, 1])\\n\\n    def design(q, rates):\\n        z = q[:, partner]\\n        return np.stack((np.ones_like(q), q, z, np.sin(rates[0]*q),\\n                         np.sin(rates[1]*z)), axis=2)\\n\\n    def profile(rates):\\n        a = design(u, rates)\\n        w = np.stack([_fit(a[:, j], v[:, j]) for j in range(4)])\\n        return a, w\\n\\n    def residual(rates):\\n        a, w = profile(rates)\\n        return (np.einsum(\'njk,jk->nj\', a, w) - v).ravel()\\n\\n    rates = least_squares(residual, [3., 2.4], bounds=([1.5, 1.5], [4.5, 4.5]),\\n                          max_nfev=80).x\\n    _, w = profile(rates)\\n    a = _fit(_affine(u), v)\\n    q = np.linalg.solve(a[1:].T, (x-a[0]).T).T\\n\\n    def forward(z):\\n        return np.einsum(\'njk,jk->nj\', design(z, rates), w)\\n\\n    def jacobian(z):\\n        j = np.zeros((len(z), 4, 4))\\n        i = np.arange(4)\\n        j[:, i, i] = w[:, 1] + rates[0]*w[:, 3]*np.cos(rates[0]*z)\\n        j[:, i, partner] = w[:, 2] + rates[1]*w[:, 4]*np.cos(rates[1]*z[:, partner])\\n        return j\\n\\n    for _ in range(14):\\n        error = forward(q)-x\\n        if not len(q) or np.max(np.abs(error)) < 1e-11:\\n            break\\n        delta = np.linalg.solve(jacobian(q), error[..., None])[..., 0]\\n        q -= np.clip(delta, -0.6, 0.6)\\n    for i in np.flatnonzero(np.max(np.abs(forward(q)-x), axis=1) > 1e-8):\\n        q[i] = least_squares(lambda z: forward(z[None, :])[0]-x[i], q[i],\\n                             jac=lambda z: jacobian(z[None, :])[0], max_nfev=60).x\\n    return q\\n\\n\\ndef _products(q):\\n    return np.column_stack((q[:, 0]*q[:, 1], q[:, 0]*q[:, 3],\\n                            q[:, 1]*q[:, 2], q[:, 2]*q[:, 3]))\\n\\n\\ndef _features(q, panel):\\n    u = panel[\'u\']\\n    c = _affine(_products(q)) @ _fit(_affine(_products(u)), panel[\'v1\'])\\n    cos = lambda z: _affine(np.cos(2.4*z))\\n    d = cos(q) @ _fit(cos(u), panel[\'v2\'])\\n    cd = (c[:, :, None]*d[:, None, :]).reshape(len(q), 16)\\n    return _affine(np.column_stack((q, c, d, cd)))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    x = np.concatenate((train[\'x\'], study[\'test\'][\'x\']), axis=0)\\n    f = _features(_calibrate(study[\'panel_a\'], x), study[\'panel_c\'])\\n    n = len(train[\'x\'])\\n    return {\'regression\': f[n:] @ _fit(f[:n], train[\'y\']),\\n            \'train_features\': f[:n], \'test_features\': f[n:]}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _ls(a, b):\\n    return np.linalg.lstsq(a, b, rcond=1e-10)[0]\\n\\n\\ndef _temporal_basis(times, trajectories):\\n    t = np.asarray(times)\\n    def basis(rates):\\n        return -np.expm1(-t[:, None]*rates)\\n    def err(rates):\\n        f = basis(rates)\\n        return (f @ _ls(f, trajectories.T) - trajectories.T).ravel()\\n    opt = least_squares(err, [1.8, .43], bounds=([.8, .03], [8., .79]), max_nfev=80)\\n    return np.linalg.qr(basis(opt.x), mode=\'reduced\')[0]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    trajectories = study[\'train\'][\'traj\']\\n    basis = _temporal_basis(study[\'eval_times\'], trajectories)\\n    return {\'basis\': basis, \'scores\': trajectories @ basis}\\n", "node_02": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    spatial=inputs[\'node_00\']; temporal=inputs[\'node_01\']\\n    y=spatial[\'regression\']\\n    weights=np.linalg.lstsq(spatial[\'train_features\'],temporal[\'scores\'],rcond=1e-10)[0]\\n    state=(spatial[\'test_features\'] @ weights) @ temporal[\'basis\'].T\\n    final=state[:,-1]\\n    return {\'y_hat\':y.copy(),\'state_pred\':state,\\n            \'diagnostics_vec\':np.array([y.mean(),y.std(),final.mean(),final.std()])}\\n"}'

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
