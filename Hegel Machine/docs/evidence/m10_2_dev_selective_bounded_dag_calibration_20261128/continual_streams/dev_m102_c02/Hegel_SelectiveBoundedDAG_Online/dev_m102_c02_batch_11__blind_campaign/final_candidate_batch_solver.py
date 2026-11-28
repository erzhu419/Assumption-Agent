"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_11_r00", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "8e7d8e53c79d8acc7c14c7bee49bef1942cb3fd87b86ff50a43d79b9dd5854c1", "node_01": "a384036848a9cb2bfc6de7b8c9b084f5e41c888a0f22db426a41dd2d180e82dd"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _calibrate_a(panel):\\n    u, v = (np.asarray(panel[k], dtype=float) for k in (\'u\', \'v\'))\\n    # Gross row corruptions are separated from ordinary channel coupling.\\n    distance = np.linalg.norm(v - u, axis=1)\\n    center = np.median(distance)\\n    mad = np.median(np.abs(distance - center))\\n    keep = distance <= center + max(1.0, 8.0 * mad)\\n    u, v = u[keep], v[keep]\\n    n, d = u.shape\\n    fixed = np.zeros((n, d, 2*d))\\n    idx = np.arange(d)\\n    fixed[:, idx, idx] = 1.\\n    fixed[:, idx, d + idx] = u\\n    opposite = np.roll(u, 2, axis=1)\\n\\n    def profile(freq):\\n        design = np.column_stack((fixed.reshape(n*d, 2*d),\\n                                  np.tanh(freq[0] * u).ravel(),\\n                                  np.sin(freq[1] * opposite).ravel()))\\n        coeff = np.linalg.lstsq(design, v.ravel(), rcond=None)[0]\\n        return design @ coeff - v.ravel(), coeff\\n\\n    opt = least_squares(lambda p: profile(p)[0], [1.8, 2.4],\\n                        bounds=([0.3, 0.5], [5., 5.]), max_nfev=120)\\n    coeff = profile(opt.x)[1]\\n    return coeff[:d], coeff[d:2*d], coeff[-2], coeff[-1], *opt.x\\n\\n\\ndef _invert_a(x, model):\\n    x = np.asarray(x, dtype=float)\\n    bias, gain, a, b, k, w = model\\n    d = len(gain)\\n    idx = np.arange(d)\\n    z = (x - bias) / gain\\n    def residual(p, y):\\n        return bias + gain*p + a*np.tanh(k*p) + b*np.sin(w*np.roll(p, 2)) - y\\n    def jacobian(p):\\n        j = np.diag(gain + a*k*(1. - np.tanh(k*p)**2))\\n        j[idx, (idx + 2) % d] += b*w*np.cos(w*np.roll(p, 2))\\n        return j\\n    for i, y in enumerate(x):\\n        z[i] = least_squares(lambda p: residual(p, y), z[i], jac=jacobian, gtol=1e-12, ftol=1e-12, xtol=1e-12, max_nfev=100).x\\n    return z\\n\\n\\ndef _affine(z):\\n    return np.column_stack((np.ones(len(z)), z))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    sensor = _calibrate_a(study[\'panel_a\'])\\n    panel = study[\'panel_b\']\\n    u, v = (np.asarray(panel[k], dtype=float) for k in (\'u\', \'v\'))\\n    operator = np.linalg.lstsq(_affine(u), v, rcond=None)[0]\\n\\n    def coordinates(x):\\n        return _affine(_invert_a(x, sensor)) @ operator\\n\\n    return {\'c\': coordinates(study[\'train\'][\'x\']),\\n            \'query\': coordinates(study[\'test\'][\'x\'])}\\n", "node_01": "\\"\\"\\"Scalar and trajectory readouts on the frozen calibrated coordinates.\\"\\"\\"\\n\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _scalar_inputs(c):\\n    paired = np.sum(c[:, :2] * c[:, 2:], axis=1)\\n    return np.column_stack((c, paired))\\n\\n\\ndef _scalar_readout(c, query, y):\\n    u, uq = _scalar_inputs(c), _scalar_inputs(query)\\n    x, xq = _affine(u), _affine(uq)\\n\\n    def design(base, z, w):\\n        s = z @ w\\n        return np.column_stack((base, s / (1.0 + s * s)))\\n\\n    def projected_residual(w):\\n        d = design(x, u, w)\\n        return d @ np.linalg.lstsq(d, y, rcond=None)[0] - y\\n\\n    linear = np.linalg.lstsq(x, y, rcond=None)[0][1:]\\n    if np.linalg.norm(linear) < 1e-10:\\n        return xq @ np.linalg.lstsq(x, y, rcond=None)[0]\\n    direction = linear / np.linalg.norm(linear)\\n    optimum = min((least_squares(projected_residual, scale * direction,\\n                                 bounds=(-5., 5.), max_nfev=160, gtol=2e-10)\\n                   for scale in (0.6, 1.2, 2.0)), key=lambda fit: fit.cost)\\n    d = design(x, u, optimum.x)\\n    beta = np.linalg.lstsq(d, y, rcond=None)[0]\\n    return design(xq, uq, optimum.x) @ beta\\n\\n\\ndef _trajectory(c, query, traj, times, intercept=True):\\n    design, new_design = (_affine(c), _affine(query)) if intercept else (c, query)\\n    beta = np.linalg.lstsq(design, traj, rcond=None)[0]\\n    fitted = design @ beta\\n    scale = max(float(np.max(np.abs(times))), np.finfo(float).tiny)\\n    t = times / scale\\n\\n    def basis(log_rates):\\n        return -np.expm1(-np.exp(log_rates[:, None]) * t)\\n\\n    def residual(log_rates):\\n        modes = basis(log_rates)\\n        return (fitted @ np.linalg.pinv(modes) @ modes - fitted).ravel()\\n\\n    log_rates = least_squares(residual, np.log([0.6, 2.7]),\\n                              bounds=(np.log([0.03, 0.25]), np.log([4., 20.])),\\n                              max_nfev=100).x\\n    modes = basis(log_rates)\\n    return new_design @ (beta @ np.linalg.pinv(modes)) @ modes\\n\\n\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    coordinates = inputs[\'node_00\']\\n    c = np.asarray(coordinates[\'c\'], dtype=float)\\n    query = np.asarray(coordinates[\'query\'], dtype=float)\\n    train = study[\'train\']\\n    y_hat = _scalar_readout(c, query, np.asarray(train[\'y\'], dtype=float))\\n    state_pred = _trajectory(c, query, np.asarray(train[\'traj\'], dtype=float),\\n                             np.asarray(study[\'eval_times\'], dtype=float))\\n    final_state = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(),\\n                                         final_state.mean(), final_state.std()])}\\n"}'

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
