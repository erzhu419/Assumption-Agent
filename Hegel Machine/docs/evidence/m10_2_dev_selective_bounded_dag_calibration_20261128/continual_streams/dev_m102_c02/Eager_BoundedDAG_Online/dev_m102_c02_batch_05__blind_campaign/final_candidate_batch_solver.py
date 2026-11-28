"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_05_r00", "depth": 4, "fanout_by_node": {"node_00": 1, "node_01": 2, "node_02": 1, "node_03": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"], "node_02": ["node_01"], "node_03": ["node_01", "node_02"]}, "max_fanout": 2, "node_code_shas": {"node_00": "7ca56815330e9326a491580714a0429053347924410908b1142de0872b963b57", "node_01": "33cd376c2c5348e1cd75ec198c5e38d2f8b82db3e98669cbbc348c4199303eea", "node_02": "f9d37d1da43c0fe301660ed1cdcf195ba0d697d8fe1431e78060b5e508e99eaf", "node_03": "b6748aa005c40c6d864a932b98a4c28a1dc3a31eb8ffcc85dab661ae26874a4c"}, "shared_fanout_nodes": ["node_01"], "sink_node_id": "node_03", "topo_order": ["node_00", "node_01", "node_02", "node_03"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import minimize_scalar\\n\\n\\ndef _sensor_fit(panel):\\n    u, v = (np.asarray(panel[key], dtype=float) for key in (\'u\', \'v\'))\\n    i, j = np.triu_indices(len(u), 1)\\n    params = []\\n    for x, y in zip(u.T, v.T):\\n        delta = x[j] - x[i]\\n        spread = np.abs(delta) > 0.15 * np.ptp(x)\\n        slope = np.median((y[j[spread]] - y[i[spread]]) / delta[spread])\\n        bias = np.median(y - slope * x)\\n        residual = y - slope * x - bias\\n        keep = np.abs(residual) <= max(6 * np.median(np.abs(residual)), 1e-10)\\n        if np.count_nonzero(keep) >= 2 and np.ptp(x[keep]) > 0:\\n            bias, slope = np.linalg.lstsq(np.column_stack((np.ones(keep.sum()), x[keep])), y[keep], rcond=None)[0]\\n        params.append((bias, slope))\\n    return np.asarray(params).T\\n\\n\\ndef _basis(x, frequency):\\n    return np.column_stack((np.ones(len(x)), x, np.sin(frequency * x)))\\n\\n\\ndef _operator(panel):\\n    u, v = (np.asarray(panel[key], dtype=float) for key in (\'u\', \'v\'))\\n\\n    def fit(frequency):\\n        design = _basis(u, frequency)\\n        coef = np.linalg.lstsq(design, v, rcond=None)[0]\\n        return np.sum(np.square(design @ coef - v)), coef\\n\\n    grid = np.linspace(0.4, 5.5, 31)\\n    best = np.argmin([fit(f)[0] for f in grid])\\n    f = minimize_scalar(lambda z: fit(z)[0], bounds=(grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]), method=\'bounded\').x\\n    return f, fit(f)[1]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    offset, gain = _sensor_fit(study[\'panel_a\'])\\n    frequency, coef = _operator(study[\'panel_b\'])\\n\\n    def coords(x):\\n        return _basis((np.asarray(x, dtype=float) - offset) / gain, frequency) @ coef\\n\\n    return {\'c\': coords(study[\'train\'][\'x\']), \'query\': coords(study[\'test\'][\'x\'])}\\n", "node_01": "\\"\\"\\"Fit the scalar readout and pass the affine trajectory projection downstream.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _interaction(x):\\n    return np.sum(x[:, :2] * x[:, 2:], axis=1)\\n\\n\\ndef _readout(x, interaction, params):\\n    z = x @ params[1:5]\\n    scale, weight = params[6:8]\\n    return params[0] + params[5] * interaction + (1.0 - weight) * z + weight * np.arctan(scale * z) / scale\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    coords = inputs[\'node_00\']\\n    c = np.asarray(coords[\'c\'], dtype=float)\\n    query = np.asarray(coords[\'query\'], dtype=float)\\n    train = study[\'train\']\\n    y = np.asarray(train[\'y\'], dtype=float)\\n    design = _affine(c)\\n    new_design = _affine(query)\\n    interaction = _interaction(c)\\n    initial = np.linalg.lstsq(np.column_stack((design, interaction)), y, rcond=None)[0]\\n    params = least_squares(\\n        lambda p: _readout(c, interaction, p) - y,\\n        np.r_[initial, 0.35, 1.0],\\n        bounds=(np.r_[np.full(6, -np.inf), 0.015, 0.0],\\n                np.r_[np.full(6, np.inf), 1.5, 5.0]),\\n        max_nfev=200,\\n    ).x\\n    beta = np.linalg.lstsq(design, np.asarray(train[\'traj\'], dtype=float), rcond=None)[0]\\n    times = np.asarray(study[\'eval_times\'], dtype=float)\\n    t = times / max(float(np.max(np.abs(times))), np.finfo(float).tiny)\\n    return {\'y_hat\': _readout(query, _interaction(query), params),\\n            \'fitted\': design @ beta, \'beta\': beta, \'new_design\': new_design, \'t\': t}\\n", "node_02": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    stage = inputs[\'node_01\']\\n    beta = np.asarray(stage[\'beta\'], dtype=float)\\n    fitted = np.asarray(stage[\'fitted\'], dtype=float)\\n    design = np.asarray(stage[\'new_design\'], dtype=float)\\n    _, _, modes = np.linalg.svd(fitted, full_matrices=False)\\n    basis = modes[:2].T\\n    return {\'state_pred\': (design @ (beta @ basis)) @ basis.T}\\n", "node_03": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    y_hat = inputs[\'node_01\'][\'y_hat\']\\n    state_pred = inputs[\'node_02\'][\'state_pred\']\\n    last = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}\\n"}'

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
