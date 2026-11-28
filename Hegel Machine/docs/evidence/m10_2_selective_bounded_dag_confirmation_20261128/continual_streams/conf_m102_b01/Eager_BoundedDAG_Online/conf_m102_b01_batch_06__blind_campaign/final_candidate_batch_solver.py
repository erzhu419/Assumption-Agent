"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_07", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "79d3bea3ad0d0b8c79be767a5be48b55d3c265eb798e6ad1e1ed3607b6821ff6", "node_01": "ba88233a698539239dea50fb6c4cfaab655a31b947017bb2bf3abe584f0ef0eb"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _aff(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _terms(u, j):\\n    a, b = u[:, j], u[:, (j + 1) % u.shape[1]]\\n    return np.column_stack((np.ones(len(a)), a, b, a*b, a**3))\\n\\n\\ndef _fit_panel_a(u, v):\\n    # Whole-row weights prevent a corrupted panel observation from leaking\\n    # into the fit of any channel.\\n    designs = [_terms(u, j) for j in range(u.shape[1])]\\n    center = np.median(v, axis=0)\\n    spread = np.maximum(np.median(np.abs(v - center), axis=0), 1e-8)\\n    score = np.max(np.abs(v - center) / spread, axis=1)\\n    keep = score <= max(6., 2.5*np.median(score))\\n    if keep.sum() < len(u)//2:\\n        keep = np.ones(len(u), dtype=bool)\\n    for _ in range(8):\\n        coef = np.array([np.linalg.lstsq(d[keep], v[keep, j], rcond=None)[0]\\n                         for j, d in enumerate(designs)])\\n        err = np.linalg.norm(np.column_stack([d @ c for d, c in zip(designs, coef)]) - v, axis=1)\\n        new = err <= max(0.001, 4*np.median(err))\\n        if np.array_equal(keep, new) or new.sum() < len(u)//2:\\n            break\\n        keep = new\\n    return coef\\n\\n\\ndef _inverse(x, c):\\n    p = x.shape[1]\\n    j = np.arange(p)\\n    k = (j + 1) % p\\n    a = np.diag(c[:, 1]); a[j, k] += c[:, 2]\\n    u = np.linalg.solve(a, (x - c[:, 0]).T).T\\n\\n    def residual(z):\\n        return c[:, 0] + c[:, 1]*z + c[:, 2]*z[:, k] + c[:, 3]*z*z[:, k] + c[:, 4]*z**3 - x\\n\\n    for _ in range(40):\\n        r = residual(u)\\n        if np.max(np.abs(r), initial=0.0) < 1e-11:\\n            break\\n        jac = np.zeros((len(u), p, p))\\n        jac[:, j, j] = c[:, 1] + c[:, 3]*u[:, k] + 3*c[:, 4]*u**2\\n        jac[:, j, k] += c[:, 2] + c[:, 3]*u\\n        step = np.linalg.solve(jac, r[..., None])[..., 0]\\n        baseline = np.sum(r*r, axis=1)\\n        trial = u - step\\n        for _ in range(15):\\n            bad = np.sum(residual(trial)**2, axis=1) > baseline\\n            if not np.any(bad):\\n                break\\n            step[bad] *= 0.5\\n            trial[bad] = u[bad] - step[bad]\\n        bad = np.sum(residual(trial)**2, axis=1) > baseline\\n        trial[bad] = u[bad]\\n        u = trial\\n    return u\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    s = inputs[\'__input__\']\\n    a, b = s[\'panel_a\'], s[\'panel_b\']\\n    c = _fit_panel_a(a[\'u\'], a[\'v\'])\\n    x = np.concatenate((s[\'train\'][\'x\'], s[\'test\'][\'x\']), axis=0)\\n    bmap = np.linalg.lstsq(_aff(b[\'u\']), b[\'v\'], rcond=None)[0]\\n    return {\'w\': _aff(_inverse(x, c)) @ bmap}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _aff(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _ls(x, y):\\n    return np.linalg.lstsq(x, y, rcond=None)[0]\\n\\n\\ndef _pairs(x):\\n    return x[:, [0, 0, 1, 2]] * x[:, [1, 3, 2, 3]]\\n\\n\\ndef _calibrate(u, v, x, transform, start, lower, upper):\\n    def residual(params):\\n        d = _aff(transform(u, params))\\n        return (d @ _ls(d, v) - v).ravel()\\n    p = least_squares(residual, start, bounds=(lower, upper), max_nfev=140).x\\n    d = _aff(transform(u, p))\\n    return _aff(transform(x, p)) @ _ls(d, v)\\n\\n\\ndef _components(s, w):\\n    b, c = s[\'panel_b\'], s[\'panel_c\']\\n    affine = _ls(_aff(b[\'u\']), b[\'v\'])\\n    u = (w - affine[0]) @ np.linalg.pinv(affine[1:])\\n    def wave(x, p):\\n        return x + p[1] * np.sin(p[0] * x)\\n    def convex(x, p):\\n        return x + p[1] * (np.logaddexp(p[0]*x, -p[0]*x) - np.log(2.0))\\n    v = _calibrate(b[\'u\'], b[\'v\'], u, wave, [2.9,-.4], [1.,-2.], [6.,0.])\\n    q = _calibrate(c[\'u\'], c[\'v2\'], u, convex, [1.1,.91], [.25,0.], [3.,4.])\\n    coef = _ls(_aff(_pairs(c[\'u\'])), c[\'v1\'])\\n    return u, v, q, coef\\n\\n\\ndef _kernel_predictions(v, q, targets, config):\\n    from scipy.linalg import solve\\n    from scipy.spatial.distance import cdist\\n\\n    n = len(targets)\\n    features = np.column_stack((v, _pairs(v), q))\\n    scaled = (features - features[:n].mean(axis=0)) / np.maximum(features[:n].std(axis=0), 1e-8)\\n    linear = scaled @ scaled[:n].T / scaled.shape[1]\\n    pq = scaled[:, v.shape[1]:]\\n    distance = cdist(pq, pq[:n], \'sqeuclidean\') / pq.shape[1]\\n    predictions = []\\n    for kind, rate, weight, ridge in config:\\n        if kind == \'pq\':\\n            nonlinear = np.exp(-rate * distance)\\n        else:\\n            nonlinear = (1.0 + linear)**2\\n        kernel = 1.0 + linear + weight * nonlinear\\n        coefficients = solve(kernel[:n] + ridge*np.eye(n), targets, assume_a=\'pos\')\\n        predictions.append(kernel[n:] @ coefficients)\\n    return np.stack(predictions)\\n\\n\\n_CONFIG = (\\n    (\'pq\', .08, .30, .00001),\\n    (\'pq\', .08, 3., .0003),\\n    (\'pq\', .16, 1., .00003),\\n    (\'pq\', .25, 20., .00001),\\n    (\'pq\', .25, 20., .003),\\n    (\'quad\', 0., 1., .015),\\n    (\'pq\', .12, 3., .00003),\\n    (\'pq\', .25, 1., .0001),\\n    (\'pq\', .10, 1., .0001),\\n)\\n_Y_WEIGHTS = (0.0, 0.0, 0.5164360648, 0.0, 0.0756475511, 0.407916384, 0.0, 0.0, 0.0)\\n_T_WEIGHTS = (0.0, 0.1214696665, 0.0, 0.0, 0.0, 0.4584866805, 0.0, 0.2778066778, 0.1422369752)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    w = np.asarray(inputs[\'node_00\'][\'w\'])\\n    _, v, q, _ = _components(study, w)\\n    train = study[\'train\']\\n    targets = np.column_stack((train[\'y\'], train[\'traj\']))\\n    p = _kernel_predictions(v, q, targets, _CONFIG)\\n    y = np.asarray(_Y_WEIGHTS) @ p[:, :, 0]\\n    t = np.tensordot(_T_WEIGHTS, p[:, :, 1:], axes=1)\\n    tail = t[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': t, \'diagnostics_vec\': np.array([y.mean(), y.std(), tail.mean(), tail.std()])}\\n"}'

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
