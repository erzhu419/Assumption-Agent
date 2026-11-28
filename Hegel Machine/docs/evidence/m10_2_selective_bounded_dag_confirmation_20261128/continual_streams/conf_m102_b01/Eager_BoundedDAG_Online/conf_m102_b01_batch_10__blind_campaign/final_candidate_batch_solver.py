"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_10_r00", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "a2d6d1ab30bcde89b84e52bcddfd59fd340ece2f9b7fa650cc2f1fa561ac29f6", "node_01": "9d625af9214fcdd5fc5698844ff32b4863bd98652f0d543df997d7d74898c773"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import least_squares, root\\n\\n\\ndef _aff(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _ls(a, b):\\n    return np.linalg.lstsq(a, b, rcond=None)[0]\\n\\n\\ndef _calibrate(panel, frequency):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    magnitude = np.max(np.abs(v), axis=1)\\n    keep = magnitude <= 2.5 * np.median(magnitude)\\n    if np.count_nonzero(keep) < 8:\\n        keep = np.ones(len(u), dtype=bool)\\n    n_dim = u.shape[1]\\n    partner = (np.arange(n_dim) + 2) % n_dim\\n\\n    def fit(mask):\\n        x, y = u[mask], v[mask]\\n        base = np.stack((np.ones_like(x), x), axis=-1).reshape(-1, 2)\\n        channels = np.tile(np.eye(n_dim), (len(x), 1))\\n        diagonal = (base[:, :, None] * channels[:, None, :]).reshape(len(base), -1)\\n\\n        def design(freqs):\\n            return np.column_stack((diagonal, np.sin(freqs[0] * x).ravel(), np.sin(freqs[1] * x[:, partner]).ravel()))\\n\\n        def residual(freqs):\\n            d = design(freqs)\\n            return d @ _ls(d, y.ravel()) - y.ravel()\\n\\n        start = np.clip([frequency + 0.9, frequency + 0.25], 0.6, 4.9)\\n        freqs = least_squares(residual, start, bounds=(0.5, 5.), gtol=1e-11).x\\n        coef = _ls(design(freqs), y.ravel())\\n        return coef[:n_dim], coef[n_dim:2*n_dim], coef[-2:], freqs, partner\\n\\n    for _ in range(4):\\n        pars = fit(keep)\\n        residual = np.linalg.norm(_forward(u, pars) - v, axis=1)\\n        new_keep = residual <= max(0.015, 5 * np.median(residual))\\n        if np.count_nonzero(new_keep) < 8 or np.array_equal(keep, new_keep):\\n            return pars\\n        keep = new_keep\\n    return fit(keep)\\n\\n\\ndef _forward(u, pars):\\n    offset, slope, amp, freq, partner = pars\\n    return offset + slope*u + amp[0]*np.sin(freq[0]*u) + amp[1]*np.sin(freq[1]*u[..., partner])\\n\\n\\ndef _invert(x, pars):\\n    offset, slope, amp, freq, partner = pars\\n    initial = (x - offset) / slope\\n    z = initial.copy()\\n    for _ in range(32):\\n        residual = _forward(z, pars) - x\\n        if not residual.size or np.max(np.abs(residual)) < 1e-10:\\n            return z\\n        z -= residual / slope\\n\\n    def jac(t):\\n        j = np.diag(slope + amp[0]*freq[0]*np.cos(freq[0]*t))\\n        j[np.arange(len(t)), partner] += amp[1]*freq[1]*np.cos(freq[1]*t[partner])\\n        return j\\n\\n    for i in np.flatnonzero(np.max(np.abs(_forward(z, pars)-x), axis=1) >= 1e-10):\\n        def fun(t):\\n            return _forward(t, pars) - x[i]\\n        fit = root(fun, z[i], jac=jac, tol=1e-10)\\n        if np.max(np.abs(fit.fun)) > 1e-8:\\n            fit = least_squares(fun, fit.x, jac=jac, gtol=1e-12)\\n        z[i] = fit.x\\n    return z\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    x = np.concatenate((study[\'train\'][\'x\'], study[\'test\'][\'x\']), axis=0)\\n    params = _calibrate(study[\'panel_a\'], float(study[\'telemetry_vec\'][0]))\\n    u = _invert(x, params)\\n    b = study[\'panel_b\']\\n    return {\'w\': _aff(u) @ _ls(_aff(b[\'u\']), b[\'v\'])}\\n", "node_01": "import itertools\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _aff(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _ls(x, y):\\n    return np.linalg.lstsq(x, y, rcond=None)[0]\\n\\n\\ndef _calibrate(u, v, x, warp, starts, bounds):\\n    def residual(p):\\n        z = _aff(warp(u, p))\\n        return (z @ _ls(z, v) - v).ravel()\\n\\n    fit = min((least_squares(residual, p, bounds=bounds, max_nfev=120)\\n               for p in starts), key=lambda fit: fit.cost)\\n    z = _aff(warp(u, fit.x))\\n    return _aff(warp(x, fit.x)) @ _ls(z, v), np.mean(np.abs(residual(fit.x)))\\n\\n\\ndef _cross(x):\\n    return _aff(np.column_stack([x[:, i] * x[:, j] for i, j in\\n                                 ((0, 1), (0, 3), (1, 2), (2, 3))]))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    w = np.asarray(inputs[\'node_00\'][\'w\'], dtype=float)\\n    a = study[\'panel_b\']\\n    affine = _ls(_aff(a[\'u\']), a[\'v\'])\\n    x = (w - affine[0]) @ np.linalg.pinv(affine[1:])\\n\\n    def warp(z, p):\\n        return (z**3 + p[0]*z)/(1 + p[1]*z*z)\\n\\n    lower = -0.85 / max(1., float(np.max(np.abs(a[\'u\'])))**2)\\n    b, b_error = _calibrate(a[\'u\'], a[\'v\'], x, warp,\\n                            ((.22, .22), (1.25, 1.25), (4., .2),\\n                             (.1, max(-.2, .9*lower))),\\n                            ([-20., lower], [40., 30.]))\\n    c = study[\'panel_c\']\\n    cz = _cross(c[\'u\'])\\n    c_coef = _ls(cz, c[\'v1\'])\\n    bilinear = _cross(x) @ c_coef\\n    c_error = np.mean(np.abs(cz @ c_coef - c[\'v1\']))\\n    quadratic = [b[:, i]*b[:, j] for i, j in\\n                 itertools.combinations_with_replacement(range(b.shape[1]), 2)]\\n    f = np.column_stack((b, bilinear, *quadratic))\\n\\n    train = study[\'train\']\\n    y = np.asarray(train[\'y\'], dtype=float).reshape(-1)\\n    target = np.column_stack((y, train[\'traj\']))\\n    n = len(y)\\n    z = _aff((f-f[:n].mean(axis=0))/np.maximum(f[:n].std(axis=0), 1e-8))\\n    t = z[:n]\\n    pen = .02*np.eye(t.shape[1]); pen[0, 0] = 0\\n    coef = np.linalg.solve(t.T @ t + pen, t.T @ target)\\n    pred = z[n:] @ coef\\n    y_hat, state_pred = pred[:, 0], pred[:, 1:]\\n    tail = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(),\\n                                         tail.mean(), tail.std()])}\\n"}'

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
