"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_04", "depth": 4, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_02": 2, "node_03": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00", "node_02"], "node_02": ["node_00"], "node_03": ["node_01", "node_02"]}, "max_fanout": 2, "node_code_shas": {"node_00": "1e01e8c32e18862a0d89127f66534e42b3ba56545b0fa01893816223887fa287", "node_01": "8fd1bb58f029a73c86ac13d93c71aec41e7dd0ec6757ddac39123e3d04a00ba9", "node_02": "d6e539ce7c4268c224476780f6252ce46e35bcbf5595e82b6682a58c83aa1025", "node_03": "1548fb750ad03eb0b8f5954ccc410a7c0e59fb309e2abfca9b051d5fa5c4e9a3"}, "shared_fanout_nodes": ["node_00", "node_02"], "sink_node_id": "node_03", "topo_order": ["node_00", "node_02", "node_01", "node_03"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import minimize_scalar\\nfrom scipy.spatial.distance import cdist\\n\\n\\ndef _calibration(panel):\\n    u, v = map(np.asarray, (panel[\'u\'], panel[\'v\']))\\n    i, j = np.triu_indices(len(u), 1)\\n    du = u[i] - u[j]\\n    slopes = np.median((v[i] - v[j]) / np.where(abs(du) > 1e-12, du, np.inf), axis=0)\\n    offsets = np.median(v - u * slopes, axis=0)\\n    for _ in range(3):\\n        r = np.linalg.norm(v - u * slopes - offsets, axis=1)\\n        med = np.median(r)\\n        keep = r <= max(6 * med, 0.0005)\\n        if keep.sum() < 5:\\n            break\\n        for k in range(u.shape[1]):\\n            slopes[k], offsets[k] = np.linalg.lstsq(np.c_[u[keep, k], np.ones(keep.sum())], v[keep, k], rcond=None)[0]\\n    return np.where(abs(slopes) > 1e-6, slopes, 1.), offsets\\n\\n\\ndef _kernel(x, q, slope):\\n    i, j = np.triu_indices(x.shape[1])\\n    return (1. + x @ q.T + .30 * ((x[:, i] * x[:, j]) @ (q[:, i] * q[:, j]).T)\\n            + 1.5 * np.exp(-cdist(x, q, \'sqeuclidean\') / 6.)\\n            + 1.5 * np.exp(-cdist(x / slope, q / slope, \'sqeuclidean\') / 6.))\\n\\n\\ndef _features(z, kind, k):\\n    if kind == \'tanh\':\\n        return np.c_[np.ones(len(z)), np.tanh(k*z)]\\n    return np.c_[np.ones(len(z)), z, z*z/(1. + k*z*z)]\\n\\n\\ndef _operator(u, v, kind, bounds):\\n    def loss(k):\\n        f = _features(u, kind, k)\\n        U, _, _ = np.linalg.svd(f, full_matrices=False)\\n        e = v - U @ (U.T @ v)\\n        return np.mean((e / np.maximum(1. - (U*U).sum(axis=1, keepdims=True), 1e-8))**2)\\n    k = minimize_scalar(loss, bounds=bounds, method=\'bounded\').x\\n    w = np.linalg.lstsq(_features(u, kind, k), v, rcond=None)[0]\\n    return lambda z: _features(z, kind, k) @ w\\n\\n\\ndef _quadratic(z):\\n    i, j = np.triu_indices(z.shape[1])\\n    return np.c_[z, z[:, i]*z[:, j]]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    x = np.asarray(train[\'x\'], dtype=float)\\n    q = np.asarray(study[\'test\'][\'x\'], dtype=float)\\n    y, t = map(np.asarray, (train[\'y\'], train[\'traj\']))\\n    slope, offset = _calibration(study[\'panel_a\'])\\n    b, c = study[\'panel_b\'], study[\'panel_c\']\\n    u = b[\'u\']\\n    w = np.linalg.lstsq(np.c_[np.ones(len(u)), u], b[\'v\'], rcond=None)[0]\\n    bmap = lambda z: z @ w[1:] + w[0]\\n    c1 = _operator(c[\'u\'], c[\'v1\'], \'tanh\', (.3, 3.))\\n    c2 = _operator(c[\'u\'], c[\'v2\'], \'rational\', (.015, .8))\\n    sy, qy = (c1(bmap((z-offset)/slope)) for z in (x, q))\\n    st, qt = (_quadratic(c2(bmap(z*slope+offset))) for z in (x, q))\\n    K, Kq = _kernel(x, x, slope), _kernel(q, x, slope)\\n    y_mean = y.mean(axis=0)\\n    y_centered = y - y_mean\\n    _, _, modes = np.linalg.svd(t, full_matrices=False)\\n    modes = modes[:2]\\n    return {\'K\': K, \'Kq\': Kq, \'sy\': sy, \'qy\': qy, \'st\': st, \'qt\': qt,\\n            \'y_mean\': np.asarray(y_mean), \'y_centered\': y_centered,\\n            \'modes\': modes, \'scores\': t @ modes.T}\\n", "node_01": "import numpy as np\\nfrom scipy.linalg import solve\\nfrom scipy.spatial.distance import cdist\\n\\n\\ndef _center(k, q):\\n    m = k.mean(axis=0)\\n    g = m.mean()\\n    return k - m - m[:, None] + g, q - m - q.mean(axis=1, keepdims=True) + g\\n\\n\\ndef _standardize(x, q):\\n    m = x.mean(axis=0)\\n    s = np.maximum(x.std(axis=0), 1e-10)\\n    return (x - m) / s, (q - m) / s\\n\\n\\ndef _poly(x, q, degree):\\n    d = x.shape[1]\\n    return _center((1 + (x @ x.T) / d) ** degree,\\n                   (1 + (q @ x.T) / d) ** degree)\\n\\n\\ndef _radial(dx, dq, width):\\n    return _center(np.exp(-dx / width), np.exp(-dq / width))\\n\\n\\ndef _distances(x, q):\\n    return cdist(x, x, \'sqeuclidean\'), cdist(q, x, \'sqeuclidean\')\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    p = inputs[\'node_00\']\\n    state = inputs[\'node_02\'][\'state_pred\']\\n    k, q = _center(p[\'K\'], p[\'Kq\'])\\n    values, vectors = np.linalg.eigh(k)\\n    d = p[\'sy\'].shape[1]\\n    rank = min(d, len(values))\\n    root = np.sqrt(np.maximum(values[-rank:], 1e-10))\\n    vectors = vectors[:, -rank:]\\n    x, xq = _standardize(vectors * root, (q @ vectors) / root)\\n    t, tq = _standardize(p[\'st\'][:, :d], p[\'qt\'][:, :d])\\n    s, sq = _standardize(p[\'sy\'], p[\'qy\'])\\n    a, aq = _standardize(p[\'scores\'], state @ p[\'modes\'].T)\\n    dt, dtq = _distances(t, tq)\\n    for w, (kt, qt) in (\\n        (.0003852, _poly(x, xq, 3)),\\n        (.0561139, _poly(x, xq, 4)),\\n        (.6507786, _radial(*_distances(x, xq), 8.)),\\n        (.0047207, _radial(dt, dtq, 4.)),\\n        (.9814442, _radial(dt, dtq, 8.)),\\n        (.084672, _poly(t, tq, 2)),\\n        (.0009055, _poly(s, sq, 3)),\\n        (.0085005, _poly(a, aq, 1)),\\n    ):\\n        k += w * kt\\n        q += w * qt\\n    alpha = solve(k + 4.7e-6 * np.eye(len(k)), p[\'y_centered\'], assume_a=\'pos\')\\n    return {\'y_hat\': np.asarray(p[\'y_mean\'] + q @ alpha)}\\n", "node_02": "import numpy as np\\nfrom scipy.linalg import solve\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    p = inputs[\'node_00\']\\n    st, qt = p[\'st\'], p[\'qt\']\\n    K = p[\'K\'] + st @ st.T\\n    Kq = p[\'Kq\'] + qt @ st.T\\n    target = p[\'scores\']\\n    mu = target.mean(axis=0)\\n    scores = Kq @ solve(K + .001*np.eye(len(K)), target - mu, assume_a=\'pos\') + mu\\n    return {\'state_pred\': scores @ p[\'modes\']}\\n", "node_03": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    yh = inputs[\'node_01\'][\'y_hat\']\\n    traj = inputs[\'node_02\'][\'state_pred\']\\n    return {\'y_hat\': yh, \'state_pred\': traj,\\n            \'diagnostics_vec\': np.array([yh.mean(), yh.std(), traj[:, -1].mean(), traj[:, -1].std()])}\\n"}'

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
