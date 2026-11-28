"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_15", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__"], "node_02": ["node_00", "node_01"]}, "max_fanout": 1, "node_code_shas": {"node_00": "557ca0b4004873567c25c886cdc313b83efe0796b4cdea9c218bec49d2bb4362", "node_01": "2f565c2ff5add3a4f117fa7e3c3893f1be39a4ba9e9c656b77bd7c83f18cc2e1", "node_02": "af0fc254511c03158325f8c23fe3b3b595193997c58ccdd811cda004a142aed7"}, "shared_fanout_nodes": [], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Fit the spatial calibration chain and construct the bilinear readout design.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares, minimize_scalar\\n\\n\\ndef _affine(u):\\n    return np.column_stack((np.ones(len(u)), u))\\n\\n\\ndef _calibrate(panel, measured):\\n    u, v = (np.asarray(panel[k], dtype=float) for k in (\'u\', \'v\'))\\n    result = np.empty_like(measured, dtype=float)\\n    for j in range(u.shape[1]):\\n        d = _affine(u[:, j])\\n        y = v[:, j]\\n        keep = np.ones(len(u), dtype=bool)\\n        for _ in range(8):\\n            coef = np.linalg.lstsq(d[keep], y[keep], rcond=None)[0]\\n            residual = np.abs(y - d @ coef)\\n            updated = residual <= max(0.0006, 4 * np.median(residual))\\n            if updated.sum() < 3 or np.array_equal(updated, keep):\\n                break\\n            keep = updated\\n        coef = np.linalg.lstsq(d[keep], y[keep], rcond=None)[0]\\n        result[:, j] = (measured[:, j] - coef[0]) / coef[1]\\n    return result\\n\\n\\ndef _profile(d, target):\\n    coef = np.linalg.lstsq(d, target, rcond=None)[0]\\n    return (d @ coef - target).ravel(), coef\\n\\n\\ndef _panel_b(panel, x):\\n    u, v = (np.asarray(panel[k], dtype=float) for k in (\'u\', \'v\'))\\n\\n    def design(q, freq):\\n        return _affine(1 - np.cos(freq * q))\\n\\n    def objective(freq):\\n        r, _ = _profile(design(u, freq), v)\\n        return r @ r\\n\\n    freq = minimize_scalar(objective, bounds=(0.5, 4.0), method=\'bounded\',\\n                           options={\'xatol\': 1e-10}).x\\n    return design(x, freq) @ _profile(design(u, freq), v)[1]\\n\\n\\ndef _panel_c(panel, b):\\n    u, v = (np.asarray(panel[k], dtype=float) for k in (\'u\', \'v1\'))\\n\\n    def design(q, p):\\n        a = np.abs(q)\\n        return _affine(q * a / (1 + np.exp(p[0]) * a) + p[1] * q**3 + p[2] * q)\\n\\n    params = least_squares(lambda p: _profile(design(u, p), v)[0],\\n                           [-0.42, -0.02, -1.08], max_nfev=200).x\\n    return design(b, params) @ _profile(design(u, params), v)[1]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = np.asarray(study[\'train\'][\'x\'], dtype=float)\\n    test = np.asarray(study[\'test\'][\'x\'], dtype=float)\\n    n = len(train)\\n    x = _calibrate(study[\'panel_a\'], np.concatenate((train, test)))\\n    b = _panel_b(study[\'panel_b\'], x)\\n    c = _panel_c(study[\'panel_c\'], b)\\n    sx, sc = x[:n].std(axis=0), c[:n].std(axis=0)\\n    x /= np.where(sx > 1e-12, sx, 1.0)\\n    c /= np.where(sc > 1e-12, sc, 1.0)\\n    interaction = (x[:, :, None] * c[:, None, :]).reshape(len(x), -1)\\n    d = _affine(np.column_stack((x, c, interaction)))\\n    return {\'train_design\': d[:n], \'test_design\': d[n:]}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    t = np.asarray(study[\'eval_times\'])\\n    traj = np.asarray(train[\'traj\'])\\n\\n    def time_basis(rates):\\n        return np.column_stack((-np.expm1(-rates[0]*t), t/(1+rates[1]*t)))\\n\\n    def time_residual(rates):\\n        basis = time_basis(rates)\\n        return (basis @ np.linalg.lstsq(basis, traj.T, rcond=None)[0] - traj.T).ravel()\\n\\n    rates = least_squares(time_residual, [1.8, .4], bounds=([.2, .05], [5., 2.])).x\\n    time = time_basis(rates)\\n    target = np.column_stack((train[\'y\'], np.linalg.lstsq(time, traj.T, rcond=None)[0].T))\\n    return {\'time\': time, \'target\': target}\\n", "node_02": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    spatial = inputs[\'node_00\']\\n    temporal = inputs[\'node_01\']\\n    a = spatial[\'train_design\']\\n    target = temporal[\'target\']\\n    time = temporal[\'time\']\\n    w = np.linalg.solve(a.T@a+.003*np.eye(a.shape[1]), a.T@target)\\n    pred = spatial[\'test_design\']@w\\n    y, state = pred[:, 0], pred[:, 1:]@time.T\\n    return dict(y_hat=y, state_pred=state, diagnostics_vec=np.array(\\n        [y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()]))\\n"}'

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
