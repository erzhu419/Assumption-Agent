"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_17", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_02": 1, "node_03": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"], "node_02": ["__input__"], "node_03": ["node_00", "node_01", "node_02"]}, "max_fanout": 2, "node_code_shas": {"node_00": "7343a733d743f0cbc92df115b5a38d5c9eabe58aaaa7c7d5c702d426f5fe2a4c", "node_01": "55cc01ccda7bbb89e46f0a001fd4f0d7f1ceee42dc8fbe9c824b2e64ecf93412", "node_02": "ebaf975ca801296e9aac40499a14456281beddbd24a3b6f32d426ccdfba2de8f", "node_03": "dbbf677e7850e6fa3d1e0babe78691aed961d406854cec412cd51e9c073820cf"}, "shared_fanout_nodes": ["node_00"], "sink_node_id": "node_03", "topo_order": ["node_00", "node_02", "node_01", "node_03"]}'
_NODE_SOURCES = '{"node_00": "\\"\\"\\"Recover latent study coordinates from the three reference panels.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import minimize_scalar\\n\\n\\ndef _calibrate(panel, measured):\\n    u, v = (np.asarray(panel[k], dtype=float) for k in (\'u\', \'v\'))\\n    affine = np.column_stack((np.ones(len(u)), u))\\n    residual = np.linalg.norm(affine @ np.linalg.lstsq(affine, v, rcond=None)[0] - v, axis=1)\\n    keep = residual < max(0.05, 4 * np.median(residual))\\n    if keep.sum() >= 8:\\n        u, v = u[keep], v[keep]\\n    d = u.shape[1]\\n    opposite = (np.arange(d) + d // 2) % d\\n\\n    def profile(freq):\\n        s = np.sin(freq*u)\\n        coefs = np.array([np.linalg.lstsq(np.column_stack((np.ones(len(u)), u[:, j], s[:, j], s[:, opposite[j]])), v[:, j], rcond=None)[0] for j in range(d)]).T\\n        err = coefs[0] + u*coefs[1] + s*coefs[2] + s[:, opposite]*coefs[3] - v\\n        return np.sum(err**2), coefs\\n\\n    freq = minimize_scalar(lambda f: profile(f)[0], bounds=(1.6, 3.2), method=\'bounded\').x\\n    offset, gain, self_amp, cross_amp = profile(freq)[1]\\n    x = (measured - offset)/gain\\n    eye = np.eye(d)\\n    for _ in range(35):\\n        sine = np.sin(freq*x)\\n        cos = freq*np.cos(freq*x)\\n        err = offset + gain*x + self_amp*sine + cross_amp*sine[:, opposite] - measured\\n        if np.max(np.abs(err)) < 1e-10:\\n            break\\n        jac = (gain + self_amp*cos)[:, :, None]*eye + (cross_amp*cos[:, opposite])[:, :, None]*eye[opposite]\\n        delta = np.linalg.solve(jac, err[..., None])[..., 0]\\n        x -= np.clip(delta, -.25, .25)\\n    return x\\n\\n\\ndef _cosine_map(panel, x):\\n    u, v = (np.asarray(panel[k], dtype=float) for k in (\'u\', \'v\'))\\n\\n    def design(z, freq):\\n        return np.column_stack((np.ones(len(z)), np.cos(freq*z)))\\n\\n    def profile(freq):\\n        h = design(u, freq)\\n        coef = np.linalg.lstsq(h, v, rcond=None)[0]\\n        return np.sum((h @ coef - v)**2), coef\\n\\n    freq = minimize_scalar(lambda f: profile(f)[0], bounds=(1.8, 3.4), method=\'bounded\').x\\n    return design(x, freq) @ profile(freq)[1]\\n\\n\\ndef _quadratic_map(panel, x):\\n    u, v = (np.asarray(panel[k], dtype=float) for k in (\'u\', \'v1\'))\\n\\n    def design(z):\\n        # The fitted response uses only products between even and odd channels.\\n        return np.column_stack((np.ones(len(z)), z[:, 0]*z[:, 1],\\n                                z[:, 0]*z[:, 3], z[:, 2]*z[:, 1],\\n                                z[:, 2]*z[:, 3]))\\n\\n    coef = np.linalg.lstsq(design(u), v, rcond=None)[0]\\n    return design(x) @ coef\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\'][\'x\']\\n    n = len(train)\\n    x = _calibrate(study[\'panel_a\'], np.vstack((train, study[\'test\'][\'x\'])))\\n    b = _cosine_map(study[\'panel_b\'], x)\\n    c = _quadratic_map(study[\'panel_c\'], b)\\n    return dict(x_train=x[:n], x_test=x[n:], c_train=c[:n], c_test=c[n:])\\n", "node_01": "\\"\\"\\"Build a common bilinear feature basis for both frozen readouts.\\"\\"\\"\\nimport numpy as np\\n\\n\\ndef _design(x, c):\\n    cross = (x[:, :, None] * c[:, None, :]).reshape(len(x), x.shape[1] * c.shape[1])\\n    return np.column_stack((np.ones(len(x)), x, c, cross))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    coords = inputs[\'node_00\']\\n    return dict(d_train=_design(coords[\'x_train\'], coords[\'c_train\']),\\n                d_test=_design(coords[\'x_test\'], coords[\'c_test\']))\\n", "node_02": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    train = inputs[\'__input__\'][\'train\']\\n    target = np.column_stack((train[\'y\'], train[\'traj\']))\\n    modes = np.linalg.svd(train[\'traj\'], full_matrices=False)[2][:2]\\n    return dict(target=target, modes=modes)", "node_03": "\\"\\"\\"Fit separate scalar and trajectory readouts from the frozen DAG outputs.\\"\\"\\"\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _features(x, c, design):\\n    # The signed saturation acts on main effects and paired interactions.\\n    main = design[:, :1 + x.shape[1] + c.shape[1]]\\n    pairs = x * c\\n    reverse_pairs = x * np.roll(c, 1, axis=1)\\n    return np.column_stack((main, pairs)), reverse_pairs\\n\\n\\ndef _scalar(a, b, y, at, bt):\\n    scale_a = np.maximum(a.std(axis=0), 0.04)\\n    scale_a[0] = 1.0\\n    scale_b = b.std(axis=0)\\n    scale_b[scale_b < 0.04] = 1.0\\n    a, b = a / scale_a, b / scale_b\\n    at, bt = at / scale_a, bt / scale_b\\n    p, q = a.shape[1], b.shape[1]\\n    penalty = np.sqrt(0.15) * np.r_[np.full(p, 0.5), 0., np.ones(q)]\\n    penalty[0] = 0.0\\n    init = np.r_[np.linalg.lstsq(a, y, rcond=None)[0] / 2.5, 2.5, np.zeros(q)]\\n\\n    def residual(w):\\n        z = a @ w[:p]\\n        return np.r_[w[p] * (z / (1 + np.abs(z))) + b @ w[p+1:] - y, penalty * w]\\n\\n    def jacobian(w):\\n        z = a @ w[:p]\\n        den = 1 + np.abs(z)\\n        return np.vstack((np.column_stack((w[p] * a / den[:, None]**2, z / den, b)), np.diag(penalty)))\\n\\n    lower = np.full(p + 1 + q, -np.inf)\\n    upper = -lower\\n    lower[p], upper[p] = 0.15, 15.0\\n    w = least_squares(residual, init, jac=jacobian, bounds=(lower, upper), max_nfev=100).x\\n    zt = at @ w[:p]\\n    return w[p] * zt / (1 + np.abs(zt)) + bt @ w[p+1:]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    coords, basis, supervision = inputs[\'node_00\'], inputs[\'node_01\'], inputs[\'node_02\']\\n    train, test = basis[\'d_train\'], basis[\'d_test\']\\n    target, modes = supervision[\'target\'], supervision[\'modes\']\\n    a, b = _features(coords[\'x_train\'], coords[\'c_train\'], train)\\n    at, bt = _features(coords[\'x_test\'], coords[\'c_test\'], test)\\n    y_hat = _scalar(a, b, target[:, 0], at, bt)\\n    coef = np.linalg.lstsq(train, target[:, 1:] @ modes.T, rcond=None)[0]\\n    state = (test @ coef) @ modes\\n    last = state[:, -1]\\n    return dict(y_hat=y_hat, state_pred=state,\\n                diagnostics_vec=np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()]))\\n"}'

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
