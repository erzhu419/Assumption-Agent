"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_11", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "d287bf9b4e4bf70bab2c6b336e21806ae6de123290e652163728347ed4c44f27", "node_01": "3897aadc3485493706942a28d95b3581750c475bc9de1fbb4d52cb16227c584c"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Invert the coupled calibration and apply the calibrated panel-B map.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares, minimize_scalar, root\\n\\n\\ndef _ls(design, target):\\n    return np.linalg.lstsq(design, target, rcond=1e-10)[0]\\n\\n\\ndef _frequency(loss, lower, upper):\\n    grid = np.linspace(lower, upper, 51)\\n    i = int(np.argmin([loss(k) for k in grid]))\\n    return minimize_scalar(loss, method=\'bounded\', bounds=(\\n        grid[max(0, i-1)], grid[min(len(grid)-1, i+1)])).x\\n\\n\\ndef _a_design(u, k, j):\\n    p, q = u[:, j], u[:, (j+2) % 4]\\n    return np.column_stack((np.ones(len(u)), p, q, np.sin(k*p), np.sin(k*q)))\\n\\n\\ndef _inverse(panel, points):\\n    u, v = (np.asarray(panel[key], dtype=float) for key in (\'u\', \'v\'))\\n    pilot = np.column_stack((np.ones(len(u)), u, u**3, u*np.roll(u, -1, axis=1)))\\n    keep = np.arange(len(u))\\n    for _ in range(4):\\n        err = np.linalg.norm(v-pilot@_ls(pilot[keep], v[keep]), axis=1)\\n        med = np.median(err)\\n        mad = np.median(abs(err-med))\\n        selected = np.flatnonzero(err <= med+max(12*mad, 0.15))\\n        if len(selected) < pilot.shape[1]+2 or np.array_equal(selected, keep):\\n            break\\n        keep = selected\\n    u, v = u[keep], v[keep]\\n    def fit(k):\\n        designs = [_a_design(u, k, j) for j in range(4)]\\n        coeff = np.array([_ls(d, v[:, j]) for j,d in enumerate(designs)])\\n        error = np.column_stack([d@coeff[j] for j,d in enumerate(designs)])-v\\n        return np.mean(error**2), coeff\\n    k = _frequency(lambda k: fit(k)[0], 1.7, 3.5)\\n    coeff = fit(k)[1]\\n    def forward(z):\\n        q = np.roll(z, -2)\\n        return (coeff[:, 0] + coeff[:, 1]*z + coeff[:, 2]*q\\n                + coeff[:, 3]*np.sin(k*z) + coeff[:, 4]*np.sin(k*q))\\n    coords = []\\n    for point in points:\\n        start = u[np.argmin(np.sum((v-point)**2, axis=1))]\\n        fun = lambda z: forward(z)-point\\n        trial = root(fun, start)\\n        if np.linalg.norm(trial.fun) > 1e-7 or np.max(abs(trial.x)) > 1.4:\\n            trial = least_squares(fun, np.clip(start, -1.39, 1.39), bounds=(-1.4, 1.4),\\n                                  ftol=1e-11, xtol=1e-11, gtol=1e-11)\\n        coords.append(trial.x)\\n    return np.asarray(coords).reshape(-1, 4)\\n\\n\\ndef _panel_b(panel, z):\\n    u, v = (np.asarray(panel[key], dtype=float) for key in (\'u\', \'v\'))\\n    def design(x, k):\\n        return np.column_stack((np.ones(len(x)), x, np.sin(k*x)))\\n    def loo(k):\\n        d = design(u, k)\\n        q, _ = np.linalg.qr(d, mode=\'reduced\')\\n        return np.mean(abs((v-d@_ls(d, v))/np.maximum(1e-8, 1-np.sum(q*q, axis=1))[:, None]))\\n    k = _frequency(loo, 1.9, 3.8)\\n    return design(z, k)@_ls(design(u, k), v)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    x = np.asarray(study[\'train\'][\'x\'], dtype=float)\\n    query = np.asarray(study[\'test\'][\'x\'], dtype=float)\\n    z = _inverse(study[\'panel_a\'], np.concatenate((x, query)))\\n    c = _panel_b(study[\'panel_b\'], z)\\n    return {\'c\': c[:len(x)], \'query\': c[len(x):]}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _readout(c, intercept=True):\\n    return np.column_stack((_affine(c) if intercept else c,\\n                            np.sum(c[:, :2] * c[:, 2:], axis=1)))\\n\\n\\ndef _trajectory(c, query, traj, times, intercept=True):\\n    design, new_design = (_affine(c), _affine(query)) if intercept else (c, query)\\n    beta = np.linalg.lstsq(design, traj, rcond=None)[0]\\n    fitted = design @ beta\\n    scale = max(float(np.max(np.abs(times))), np.finfo(float).tiny)\\n    t = times / scale\\n\\n    def basis(log_rates):\\n        return -np.expm1(-np.exp(log_rates[:, None]) * t)\\n\\n    def residual(log_rates):\\n        modes = basis(log_rates)\\n        return (fitted @ np.linalg.pinv(modes) @ modes - fitted).ravel()\\n\\n    log_rates = least_squares(residual, np.log([0.6, 2.7]),\\n                              bounds=(np.log([0.03, 0.25]), np.log([4., 20.])),\\n                              max_nfev=100).x\\n    modes = basis(log_rates)\\n    return new_design @ (beta @ np.linalg.pinv(modes)) @ modes\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    coordinates = inputs[\'node_00\']\\n    c, query = coordinates[\'c\'], coordinates[\'query\']\\n    train = study[\'train\']\\n    beta = np.linalg.lstsq(_readout(c), np.asarray(train[\'y\'], dtype=float), rcond=None)[0]\\n    y_hat = _readout(query) @ beta\\n    state_pred = _trajectory(c, query, np.asarray(train[\'traj\'], dtype=float),\\n                             np.asarray(study[\'eval_times\'], dtype=float))\\n    last = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}\\n"}'

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
