"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_05", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__"], "node_02": ["node_00", "node_01"]}, "max_fanout": 1, "node_code_shas": {"node_00": "e3ea5db76e85c39bd5245a330ae8b46a521144d777df6845916b7ffc7fe128d0", "node_01": "1a2156e508de93b4669fc2bbec325fb6f23696e9276f601cb8ad2c0b45a8b9c7", "node_02": "1bc6c25e9865f5e3ed0287854598a75d23bbafc4dccba54ee52bd90d7c12374d"}, "shared_fanout_nodes": [], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Recover calibrated coordinates and predict the two branch readouts.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares, root\\n\\n\\ndef _ls(design, values):\\n    return np.linalg.lstsq(design, values, rcond=None)[0]\\n\\n\\ndef _intercept(features):\\n    return np.column_stack((np.ones(len(features)), features))\\n\\n\\ndef _calibration_features(z, neighbor):\\n    return _intercept(np.column_stack((z, neighbor, z * neighbor, z**3)))\\n\\n\\ndef _calibrate(panel, observed):\\n    u, v = np.asarray(panel[\'u\']), np.asarray(panel[\'v\'])\\n    width = u.shape[1]\\n    right = (np.arange(width) + 1) % width\\n    coefficients = []\\n    for j, k in enumerate(right):\\n        design = _calibration_features(u[:, j], u[:, k])\\n        y = v[:, j]\\n        robust = least_squares(lambda p: design @ p - y, _ls(design, y),\\n                               loss=\'soft_l1\', f_scale=.02).x\\n        error = design @ robust - y\\n        center = np.median(error)\\n        mad = np.median(np.abs(error - center))\\n        clean = np.abs(error - center) <= max(.025, 8 * mad)\\n        coefficients.append(_ls(design[clean], y[clean]) if clean.sum() >= 5 else robust)\\n    offset, gain, coupling, product, cubic = np.asarray(coefficients).T\\n    linear = np.diag(gain)\\n    linear[np.arange(width), right] = coupling\\n    starts = np.linalg.solve(linear, (observed - offset).T).T\\n    result = []\\n    for value, initial in zip(observed, starts):\\n        def residual(z):\\n            return offset + gain*z + (coupling + product*z)*z[right] + cubic*z**3 - value\\n\\n        def jacobian(z):\\n            jac = np.diag(gain + product*z[right] + 3*cubic*z**2)\\n            jac[np.arange(width), right] += coupling + product*z\\n            return jac\\n\\n        fit = root(residual, initial, jac=jacobian, tol=1e-10)\\n        z = fit.x\\n        if np.max(np.abs(residual(z))) > 1e-8:\\n            alternate = least_squares(residual, z, jac=jacobian).x\\n            if np.linalg.norm(residual(alternate)) < np.linalg.norm(residual(z)):\\n                z = alternate\\n        result.append(z)\\n    return np.asarray(result).reshape(observed.shape)\\n\\n\\ndef _branches(panel, z):\\n    u = np.asarray(panel[\'u\'])\\n\\n    def pairs(x):\\n        return _intercept(x * np.roll(x, -1, axis=1))\\n\\n    first = pairs(z) @ _ls(pairs(u), panel[\'v1\'])\\n    target = np.asarray(panel[\'v2\'])\\n\\n    def cosine(x, p):\\n        return _intercept(np.column_stack((x, np.cos(p[0] * x + p[1:]))))\\n\\n    def residual(p):\\n        design = cosine(u, p)\\n        return (design @ _ls(design, target) - target).ravel()\\n\\n    parameters = least_squares(residual, np.r_[1.27, np.zeros(u.shape[1])],\\n                               max_nfev=150).x\\n    second = cosine(z, parameters) @ _ls(cosine(u, parameters), target)\\n    return first, second\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    n = len(train[\'x\'])\\n    measured = np.vstack((train[\'x\'], study[\'test\'][\'x\']))\\n    calibrated = _calibrate(study[\'panel_a\'], measured)\\n    first, second = _branches(study[\'panel_c\'], calibrated)\\n    cross = second[:, :, None] * first[:, None, :]\\n    scalar = _intercept(np.column_stack((first, second, np.trace(cross, axis1=1, axis2=2))))\\n    dynamic = _intercept(np.column_stack((first, second, cross.reshape(len(measured), -1))))\\n    return {\'y_hat\': scalar[n:] @ _ls(scalar[:n], train[\'y\']),\\n            \'unprojected\': dynamic[n:] @ _ls(dynamic[:n], train[\'traj\'])}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _ls(a, b):\\n    return np.linalg.lstsq(a, b, rcond=None)[0]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    ts = np.asarray(study[\'eval_times\'])\\n    trajectories = study[\'train\'][\'traj\'][slice(None)]\\n    def basis(p):\\n        return -np.expm1(-ts[:, None]*p)\\n    def time_residual(p):\\n        h = basis(p)\\n        return (h @ _ls(h, trajectories.T) - trajectories.T).ravel()\\n    h = basis(least_squares(time_residual, [1.8,.4], bounds=([1.,.1],[3.,.95])).x)\\n    return {\'h\': h}\\n", "node_02": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    fits = inputs[\'node_00\']\\n    h = inputs[\'node_01\'][\'h\']\\n    y = fits[\'y_hat\']\\n    state = fits[\'unprojected\'] @ np.linalg.pinv(h.T) @ h.T\\n    terminal = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])}\\n"}'

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
