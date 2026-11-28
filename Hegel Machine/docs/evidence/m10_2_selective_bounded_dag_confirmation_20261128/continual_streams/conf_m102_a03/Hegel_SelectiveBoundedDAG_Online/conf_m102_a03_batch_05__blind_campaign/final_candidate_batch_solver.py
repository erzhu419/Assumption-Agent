"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_06", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"], "node_02": ["node_00", "node_01"]}, "max_fanout": 2, "node_code_shas": {"node_00": "ae3c2f3197bc2963ca1c863f431c641aa0cfb026514a1be8321ee32192440924", "node_01": "72c1589bf3d652eb8dafdb7e51d95ad9220892791e5e2a77768b675e869f8761", "node_02": "6b2aed9db8bed4ef4803a05cc4d3b369a6ae5906b5a63da862248cde58bac85e"}, "shared_fanout_nodes": ["node_00"], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Calibrate the sensor coordinates and construct the scalar and state kernels.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\nfrom scipy.spatial.distance import cdist\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _fit(a, b):\\n    return np.linalg.lstsq(a, b, rcond=None)[0]\\n\\n\\ndef _sensor_calibration(panel):\\n    u, v = (np.asarray(panel[k], dtype=float) for k in (\'u\', \'v\'))\\n    a = np.stack((np.ones_like(u), u), axis=-1)\\n    w = np.ones(len(u))\\n    for _ in range(24):\\n        co = np.column_stack([_fit(a[:, j] * w[:, None], v[:, j] * w)\\n                              for j in range(u.shape[1])])\\n        r = np.linalg.norm(v - co[0] - co[1] * u, axis=1)\\n        w = np.sqrt(np.minimum(1., 1.5 * max(np.median(r), 1e-12) / np.maximum(r, 1e-12)))\\n    return co\\n\\n\\ndef _readout_coordinates(panel, z):\\n    u, v = (np.asarray(panel[k], dtype=float) for k in (\'u\', \'v\'))\\n    d = u.shape[1]\\n\\n    def basis(x, p, trig):\\n        h = x * np.exp(p[:d]) + p[d:]\\n        if trig:\\n            return np.column_stack((_affine(x), x*x, np.sin(h)))\\n        return np.column_stack((_affine(x), np.tanh(h)))\\n\\n    candidates = []\\n    for trig in (False, True):\\n        def residual(p):\\n            f = basis(u, p, trig)\\n            return (f @ _fit(f, v) - v).ravel()\\n\\n        result = min((least_squares(residual, np.r_[np.full(d, k), np.zeros(d)],\\n                                    bounds=(-2., 2.), max_nfev=90)\\n                      for k in (-.3, .4, 1.)), key=lambda r: r.cost)\\n        f = basis(u, result.x, trig)\\n        score = np.linalg.norm(result.fun) * (1 + f.shape[1] / 20.)\\n        candidates.append((score, basis(z, result.x, trig) @ _fit(f, v)))\\n    return min(candidates, key=lambda item: item[0])[1]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    co = _sensor_calibration(study[\'panel_a\'])\\n    x, xt = (np.asarray(block[\'x\'], dtype=float) for block in (train, study[\'test\']))\\n    n = len(x)\\n    z = (np.concatenate((x, xt), axis=0) - co[0]) / co[1]\\n    w = _readout_coordinates(study[\'panel_b\'], z)\\n    w = (w - w[:n].mean(0)) / np.maximum(w[:n].std(0), 1e-10)\\n    wt, w = w[n:], w[:n]\\n    # Use sensor coordinates for additive effects that the panel map mixes.\\n    u = (z - z[:n].mean(0)) / np.maximum(z[:n].std(0), 1e-10)\\n\\n    def additive(a, b, rate):\\n        return np.exp(-rate * (a[:, None, :] - b[None, :, :])**2).mean(2)\\n\\n    def scalar(a, b, ua, ub):\\n        d = a.shape[1]\\n        return (1. + a @ b.T / d + np.exp(-.15 * cdist(a, b, \'sqeuclidean\') / d)\\n                + .3 * additive(a, b, .15)\\n                + .5 * (ua @ ub.T / ua.shape[1] + additive(ua, ub, 1.2)))\\n\\n    def state(a, b):\\n        return 1. + a @ b.T + np.exp(-1.2 * (a[:, None, :] - b[None, :, :])**2).mean(2)\\n\\n    modes = np.linalg.svd(np.asarray(train[\'traj\'], dtype=float), full_matrices=False)[2][:2]\\n    ks = state(z[:n], z[:n]) + .001 * np.eye(n)\\n    kt = state(z[n:], z[:n])\\n    return {\'readout_train\': scalar(w, w, u[:n], u[:n]) + .01 * np.eye(n),\\n            \'readout_test\': scalar(wt, w, u[n:], u[:n]),\\n            \'state_train\': np.stack([ks] * len(modes)),\\n            \'modes\': modes,\\n            \'zt\': np.concatenate([kt] * len(modes), axis=1)}\\n", "node_01": "\\"\\"\\"Fit the scalar readout and two temporal modes using the calibrated kernels.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.linalg import solve\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    train = inputs[\'__input__\'][\'train\']\\n    features = inputs[\'node_00\']\\n    readout = solve(features[\'readout_train\'], np.asarray(train[\'y\'], dtype=float), assume_a=\'pos\')\\n    traj = np.asarray(train[\'traj\'], dtype=float)\\n    modes = features[\'modes\']\\n    blocks = []\\n    for K, mode in zip(features[\'state_train\'], modes):\\n        coef = solve(K, traj @ mode, assume_a=\'pos\')\\n        blocks.append(np.outer(coef, mode))\\n    return {\'readout_coef\': readout, \'state_coef\': np.concatenate(blocks, axis=0)}\\n", "node_02": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    features = inputs[\'node_00\']\\n    params = inputs[\'node_01\']\\n    yh = features[\'readout_test\'] @ params[\'readout_coef\']\\n    sp = features[\'zt\'] @ params[\'state_coef\']\\n    diag = np.array([yh.mean(), yh.std(), sp[:, -1].mean(), sp[:, -1].std()])\\n    return {\'y_hat\': yh, \'state_pred\': sp, \'diagnostics_vec\': diag}\\n"}'

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
