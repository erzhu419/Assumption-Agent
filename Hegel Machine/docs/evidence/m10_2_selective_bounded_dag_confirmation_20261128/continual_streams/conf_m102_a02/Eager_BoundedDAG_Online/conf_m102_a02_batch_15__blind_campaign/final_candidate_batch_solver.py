"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_17", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"], "node_02": ["node_00", "node_01"]}, "max_fanout": 2, "node_code_shas": {"node_00": "f253fc254797595e6661454a2639d9661e7e441e165ab278d8a837bcdabb9345", "node_01": "aa26baf276ca32674689e030671c6e6f8a6c231f009beef1eb186a44b8638ca1", "node_02": "c511ca91569f4f92442799147e67ed8f097fbd7e5643c07b827afca8c89039dd"}, "shared_fanout_nodes": ["node_00"], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "from itertools import combinations_with_replacement\\nimport numpy as np\\n\\n\\ndef _design(x):\\n    x = np.asarray(x, dtype=float)\\n    affine = np.column_stack((np.ones(len(x)), x))\\n    quadratic = [x[:, i] * x[:, j]\\n                 for i, j in combinations_with_replacement(range(x.shape[1]), 2)]\\n    return affine, np.column_stack((affine, *quadratic))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    a, q = _design(study[\'train\'][\'x\'])\\n    a_test, q_test = _design(study[\'test\'][\'x\'])\\n    return {\'a\': a, \'q\': q, \'a_test\': a_test, \'q_test\': q_test}\\n", "node_01": "\\"\\"\\"Fit the reference-panel calibration and the two calibrated readouts.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares, minimize_scalar\\n\\n\\ndef _fit(design, target):\\n    return np.linalg.lstsq(design, target, rcond=None)[0]\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _calibrate(panel, x):\\n    u, v = (np.asarray(panel[key], dtype=float) for key in (\'u\', \'v\'))\\n    n, d = u.shape\\n    axes = np.arange(d)\\n    opposite = (axes + d // 2) % d\\n    w = u[:, opposite]\\n\\n    def design(freq):\\n        a = np.zeros((n, d, 2*d + 2))\\n        a[:, axes, axes] = 1.\\n        a[:, axes, axes+d] = u\\n        a[:, :, -2] = np.tanh(freq[0]*u)\\n        a[:, :, -1] = np.sin(freq[1]*w)\\n        return a\\n\\n    def coefficients(a, keep):\\n        return _fit(a[keep].reshape(-1, a.shape[-1]), v[keep].ravel())\\n\\n    # Screen the gross, rowwise reference faults before profiling frequency.\\n    size = np.max(np.abs(v), axis=1)\\n    med = np.median(size)\\n    mad = np.median(np.abs(size - med))\\n    keep = size <= med + 8*max(1.4826*mad, 0.1)\\n    freq = np.array([1.8, 2.4])\\n    for _ in range(6):\\n        def residual(k):\\n            a = design(k)\\n            return (a[keep] @ coefficients(a, keep) - v[keep]).ravel()\\n\\n        freq = least_squares(residual, freq, bounds=([0.7, 1.5], [3.2, 3.4]),\\n                             max_nfev=60).x\\n        a = design(freq)\\n        c = coefficients(a, keep)\\n        err = np.linalg.norm(a @ c - v, axis=1)\\n        med = np.median(err[keep])\\n        mad = np.median(np.abs(err[keep] - med))\\n        updated = err <= med + 6*max(1.4826*mad, 3e-4)\\n        if updated.sum() < max(8, n//2) or np.array_equal(updated, keep):\\n            break\\n        keep = updated\\n\\n    offset, scale = c[:d], c[d:2*d]\\n    amp_self, amp_opp = c[-2:]\\n    a, b = freq\\n\\n    def invert(row):\\n        def residual(z):\\n            return offset + scale*z + amp_self*np.tanh(a*z) + amp_opp*np.sin(b*z[opposite]) - row\\n\\n        def jac(z):\\n            j = np.diag(scale + amp_self*a*(1 - np.tanh(a*z)**2))\\n            j[axes, opposite] = amp_opp*b*np.cos(b*z[opposite])\\n            return j\\n\\n        return least_squares(residual, (row-offset)/scale, jac=jac, max_nfev=60).x\\n\\n    return np.array([invert(row) for row in x]).reshape(-1, d)\\n\\n\\ndef _panel_coordinates(panel, z):\\n    u, v = (np.asarray(panel[key], dtype=float) for key in (\'u\', \'v\'))\\n\\n    def design(x, frequency):\\n        return np.column_stack((_affine(x), np.cos(frequency*x)))\\n\\n    def loss(frequency):\\n        a = design(u, frequency)\\n        return np.mean((a @ _fit(a, v) - v)**2)\\n\\n    frequency = minimize_scalar(loss, method=\'bounded\', bounds=(2.1, 3.1),\\n                                options={\'xatol\': 1e-9}).x\\n    return design(z, frequency) @ _fit(design(u, frequency), v)\\n\\n\\ndef _readout_design(g):\\n    h = g.shape[1] // 2\\n    return np.column_stack((_affine(g), g[:, :h]*g[:, h:]))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study, raw = inputs[\'__input__\'], inputs[\'node_00\']\\n    train_x = raw[\'a\'][:, 1:]\\n    test_x = raw[\'a_test\'][:, 1:]\\n    latent = _calibrate(study[\'panel_a\'], np.vstack((train_x, test_x)))\\n    g = _panel_coordinates(study[\'panel_b\'], latent)\\n    design = _readout_design(g)\\n    train_design, test_features = design[:len(train_x)], design[len(train_x):]\\n    y = np.asarray(study[\'train\'][\'y\'], dtype=float)\\n    traj = np.asarray(study[\'train\'][\'traj\'], dtype=float)\\n    p = g.shape[1] + 1\\n    coefficients = np.zeros((design.shape[1], 1 + traj.shape[1]))\\n    coefficients[:, 0] = _fit(train_design, y)\\n    coefficients[:p, 1:] = _fit(train_design[:, :p], traj)\\n    return {\'test_features\': test_features, \'coefficients\': coefficients}\\n", "node_02": "\\"\\"\\"Apply the fitted readouts to the calibrated test design.\\"\\"\\"\\nimport numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    raw, fitted = inputs[\'node_00\'], inputs[\'node_01\']\\n    z = fitted[\'test_features\']\\n    if len(z) != len(raw[\'a_test\']):\\n        raise ValueError(\'Calibrated design must retain all test rows\')\\n    values = z @ fitted[\'coefficients\']\\n    endpoint, state = values[:, 0], values[:, 1:]\\n    last = state[:, -1]\\n    return {\'y_hat\': endpoint, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([endpoint.mean(), endpoint.std(), last.mean(), last.std()])}\\n"}'

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
