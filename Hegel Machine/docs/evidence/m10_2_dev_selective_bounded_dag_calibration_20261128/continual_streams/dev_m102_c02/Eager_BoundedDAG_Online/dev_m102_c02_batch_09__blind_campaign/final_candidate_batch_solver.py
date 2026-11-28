"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_10", "depth": 4, "fanout_by_node": {"node_00": 1, "node_01": 2, "node_02": 1, "node_03": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"], "node_02": ["node_01"], "node_03": ["node_01", "node_02"]}, "max_fanout": 2, "node_code_shas": {"node_00": "d44212f955f1acc278bfc5faf8541154f650033666520bfe0c34d53af2a0ba20", "node_01": "69c4d3f085eedea8d05dbcd97ffc59b8a7aeea5527819f2eb24ada5633ea56be", "node_02": "f7a1e3128b0f0d1dcca739b4f1ac68ad4ffc6e0a8e92660c37df1ba9c11004b9", "node_03": "b6748aa005c40c6d864a932b98a4c28a1dc3a31eb8ffcc85dab661ae26874a4c"}, "shared_fanout_nodes": ["node_01"], "sink_node_id": "node_03", "topo_order": ["node_00", "node_01", "node_02", "node_03"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Recover the shared coordinates from the two calibration panels.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _fit_sensor(u, v):\\n    # The gross faults affect entire paired calibration rows. Screen them\\n    # against a robust diagonal fit before fitting the coupled response.\\n    i, j = np.triu_indices(len(u), 1)\\n    dx = u[j] - u[i]\\n    dy = v[j] - v[i]\\n    slopes = np.full_like(dx, np.nan)\\n    np.divide(dy, dx, out=slopes, where=np.abs(dx) > 0.1 * np.ptp(u, axis=0))\\n    slope = np.nanmedian(slopes, axis=0)\\n    bias = np.median(v - u * slope, axis=0)\\n    err = np.abs(v - (bias + u * slope))\\n    keep = np.all(err <= np.maximum(8 * np.median(err, axis=0), 1e-4), axis=1)\\n    if np.count_nonzero(keep) >= 8:\\n        u, v = u[keep], v[keep]\\n    paired = np.roll(u, 2, axis=1)\\n    d = np.zeros((len(u), 4, 10))\\n    k = np.arange(4)\\n    d[:, k, k] = 1\\n    d[:, k, 4 + k] = u\\n\\n    def fit(freq, return_coeff=False):\\n        d[:, :, 8] = np.tanh(freq[0] * u)\\n        d[:, :, 9] = np.sin(freq[1] * paired)\\n        design = d.reshape(-1, 10)\\n        coeff = np.linalg.lstsq(design, v.ravel(), rcond=None)[0]\\n        if return_coeff:\\n            return coeff\\n        return design @ coeff - v.ravel()\\n\\n    freq = least_squares(fit, [1.8, 2.4], bounds=([0.5, 0.8], [4.0, 4.0])).x\\n    return freq, fit(freq, return_coeff=True)\\n\\n\\ndef _invert(x, freq, coeff):\\n    f, g = freq\\n    offset, gain, a, b = coeff[:4], coeff[4:8], coeff[8], coeff[9]\\n    shifted = x - offset\\n    z = shifted / gain\\n    # Solve the strictly increasing self-response exactly within each\\n    # paired update. The fitted cross-response is a contraction.\\n    for _ in range(240):\\n        rhs = shifted - b * np.sin(g * np.roll(z, 2, axis=1))\\n        q = rhs / gain\\n        for _ in range(8):\\n            t = np.tanh(f * q)\\n            q -= (gain * q + a * t - rhs) / (gain + a * f * (1 - t * t))\\n        delta = np.max(np.abs(q - z), initial=0.0)\\n        z = q\\n        if delta < 2e-12:\\n            break\\n    return z\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    a, b = study[\'panel_a\'], study[\'panel_b\']\\n    freq, coeff = _fit_sensor(np.asarray(a[\'u\'], dtype=float), np.asarray(a[\'v\'], dtype=float))\\n    operator = np.linalg.lstsq(_affine(np.asarray(b[\'u\'], dtype=float)),\\n                              np.asarray(b[\'v\'], dtype=float), rcond=None)[0]\\n    train = np.asarray(study[\'train\'][\'x\'], dtype=float)\\n    query = np.asarray(study[\'test\'][\'x\'], dtype=float)\\n    coords = _affine(_invert(np.concatenate((train, query)), freq, coeff)) @ operator\\n    return {\'c\': coords[:len(train)], \'query\': coords[len(train):]}\\n", "node_01": "import numpy as np\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _readout(c):\\n    return np.column_stack((_affine(c), np.sum(c[:, :2] * c[:, 2:], axis=1)))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    coords = inputs[\'node_00\']\\n    train = study[\'train\']\\n    c, query = coords[\'c\'], coords[\'query\']\\n    readout_beta = np.linalg.lstsq(_readout(c), np.asarray(train[\'y\'], dtype=float), rcond=None)[0]\\n    y_hat = _readout(query) @ readout_beta\\n    design, new_design = _affine(c), _affine(query)\\n    beta = np.linalg.lstsq(design, np.asarray(train[\'traj\'], dtype=float), rcond=None)[0]\\n    fitted = design @ beta\\n    times = np.asarray(study[\'eval_times\'], dtype=float)\\n    scale = max(float(np.max(np.abs(times))), np.finfo(float).tiny)\\n    return {\'y_hat\': y_hat, \'beta\': beta, \'fitted\': fitted,\\n            \'new_design\': new_design, \'t\': times / scale}\\n", "node_02": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    projection = inputs[\'node_01\']\\n    fitted, t = projection[\'fitted\'], projection[\'t\']\\n\\n    def basis(log_rates):\\n        return -np.expm1(-np.exp(log_rates[:, None]) * t)\\n\\n    def residual(log_rates):\\n        modes = basis(log_rates)\\n        return (fitted @ np.linalg.pinv(modes) @ modes - fitted).ravel()\\n\\n    log_rates = least_squares(residual, np.log([0.6, 2.7]),\\n                              bounds=(np.log([0.03, 0.25]), np.log([4., 20.])),\\n                              max_nfev=100).x\\n    modes = basis(log_rates)\\n    state_pred = projection[\'new_design\'] @ (projection[\'beta\'] @ np.linalg.pinv(modes)) @ modes\\n    return {\'state_pred\': state_pred}\\n", "node_03": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    y_hat = inputs[\'node_01\'][\'y_hat\']\\n    state_pred = inputs[\'node_02\'][\'state_pred\']\\n    last = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}\\n"}'

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
