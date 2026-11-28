"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_05", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "d87c2232b13f046bed3c1c9cc5fd328353d1383eda24d91b07b21c34e67a9c5a", "node_01": "3897aadc3485493706942a28d95b3581750c475bc9de1fbb4d52cb16227c584c"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Recover readout coordinates from the two paired calibration panels.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares, minimize_scalar\\n\\n\\ndef _sensor(u, v):\\n    nxt = np.roll(u, -1, axis=1)\\n    # Gross contamination affects complete calibration rows.\\n    size = np.max(np.abs(v), axis=1)\\n    center = np.median(size)\\n    spread = np.median(np.abs(size - center))\\n    keep = size <= center + 9 * max(spread, 1e-8)\\n    if keep.sum() < 8:\\n        keep = np.ones(len(u), dtype=bool)\\n\\n    def design(j, k):\\n        x = u[:, j]\\n        return np.column_stack((np.ones(len(u)), x, nxt[:, j], x*nxt[:, j], (np.sin(k*x)-k*x)/k**3))\\n\\n    def fit(k):\\n        ds = [design(j, k)[keep] for j in range(4)]\\n        ws = [np.linalg.lstsq(d, v[keep, j], rcond=None)[0] for j, d in enumerate(ds)]\\n        return sum(np.sum((d@w-v[keep, j])**2) for j, (d, w) in enumerate(zip(ds, ws))), np.array(ws)\\n\\n    k = minimize_scalar(lambda k: fit(k)[0], bounds=(.06, 1.7), method=\'bounded\').x\\n    return k, fit(k)[1]\\n\\n\\ndef _invert(x, k, w):\\n    def forward(z):\\n        nxt = np.roll(z, -1)\\n        return w[:, 0] + w[:, 1]*z + w[:, 2]*nxt + w[:, 3]*z*nxt + w[:, 4]*(np.sin(k*z)-k*z)/k**3\\n\\n    def jac(z):\\n        a = np.diag(w[:, 1]+w[:, 3]*np.roll(z, -1)+w[:, 4]*(np.cos(k*z)-1)/k**2)\\n        j = np.arange(4)\\n        a[j, (j+1)%4] = w[:, 2]+w[:, 3]*z\\n        return a\\n\\n    return np.array([least_squares(lambda z: forward(z)-row, row, jac=jac, gtol=1e-11, max_nfev=80).x for row in x]).reshape(-1, 4)\\n\\n\\ndef _second(u, v, x):\\n    def design(a, scale):\\n        return np.column_stack((np.ones(len(a)), a, np.tanh(scale*a)))\\n\\n    def fit(scale):\\n        d = design(u, scale)\\n        w = np.linalg.lstsq(d, v, rcond=None)[0]\\n        return np.sum((d@w-v)**2), w\\n\\n    scale = minimize_scalar(lambda scale: fit(scale)[0], bounds=(.35, 3.), method=\'bounded\').x\\n    return design(x, scale) @ fit(scale)[1]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    a = study[\'panel_a\']\\n    b = study[\'panel_b\']\\n    train = np.asarray(study[\'train\'][\'x\'], dtype=float)\\n    query = np.asarray(study[\'test\'][\'x\'], dtype=float)\\n    k, w = _sensor(np.asarray(a[\'u\'], dtype=float), np.asarray(a[\'v\'], dtype=float))\\n    latent = _invert(np.concatenate((train, query)), k, w)\\n    coord = _second(np.asarray(b[\'u\'], dtype=float), np.asarray(b[\'v\'], dtype=float), latent)\\n    return {\'c\': coord[:len(train)], \'query\': coord[len(train):]}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _readout(c, intercept=True):\\n    return np.column_stack((_affine(c) if intercept else c,\\n                            np.sum(c[:, :2] * c[:, 2:], axis=1)))\\n\\n\\ndef _trajectory(c, query, traj, times, intercept=True):\\n    design, new_design = (_affine(c), _affine(query)) if intercept else (c, query)\\n    beta = np.linalg.lstsq(design, traj, rcond=None)[0]\\n    fitted = design @ beta\\n    scale = max(float(np.max(np.abs(times))), np.finfo(float).tiny)\\n    t = times / scale\\n\\n    def basis(log_rates):\\n        return -np.expm1(-np.exp(log_rates[:, None]) * t)\\n\\n    def residual(log_rates):\\n        modes = basis(log_rates)\\n        return (fitted @ np.linalg.pinv(modes) @ modes - fitted).ravel()\\n\\n    log_rates = least_squares(residual, np.log([0.6, 2.7]),\\n                              bounds=(np.log([0.03, 0.25]), np.log([4., 20.])),\\n                              max_nfev=100).x\\n    modes = basis(log_rates)\\n    return new_design @ (beta @ np.linalg.pinv(modes)) @ modes\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    coordinates = inputs[\'node_00\']\\n    c, query = coordinates[\'c\'], coordinates[\'query\']\\n    train = study[\'train\']\\n    beta = np.linalg.lstsq(_readout(c), np.asarray(train[\'y\'], dtype=float), rcond=None)[0]\\n    y_hat = _readout(query) @ beta\\n    state_pred = _trajectory(c, query, np.asarray(train[\'traj\'], dtype=float),\\n                             np.asarray(study[\'eval_times\'], dtype=float))\\n    last = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}\\n"}'

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
