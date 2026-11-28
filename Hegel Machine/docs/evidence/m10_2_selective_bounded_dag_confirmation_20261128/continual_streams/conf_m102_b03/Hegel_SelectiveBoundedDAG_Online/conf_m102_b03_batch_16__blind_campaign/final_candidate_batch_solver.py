"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_11_r01", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "378555358d0499b1ca44c01e973c10255b69c6ac01b1d2986da8cc7585c2aa4a", "node_01": "98d65092b7379a4648e553322eb1f55312f0c0b3688603354fd848360ba0581d"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import minimize_scalar\\n\\n\\ndef _ones(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _ls(x, y):\\n    return np.linalg.lstsq(x, y, rcond=None)[0]\\n\\n\\ndef _calibration(panel, x):\\n    u, v = (np.asarray(panel[k], dtype=float) for k in (\'u\', \'v\'))\\n    keep = abs(v-u-np.median(v-u, axis=0)) < 1.5\\n    axis = np.arange(u.shape[1])\\n    other = (axis+2) % len(axis)\\n    w = u[:, other]\\n    base = np.concatenate((np.broadcast_to(np.eye(len(axis)), (len(u), len(axis), len(axis))),\\n                           u[:, :, None]*np.eye(len(axis))), axis=2)\\n\\n    def fit(freq):\\n        design = np.concatenate((base, np.stack((w, np.sin(freq*u), np.sin(freq*w)), axis=2)), axis=2)[keep]\\n        coef = _ls(design, v[keep])\\n        err = design @ coef - v[keep]\\n        return err @ err, coef\\n\\n    freq = minimize_scalar(lambda f: fit(f)[0], bounds=(1.7, 2.65), method=\'bounded\').x\\n    _, coeff = fit(freq)\\n    offset, scale = np.split(coeff[:-3], 2)\\n    coupling, own_sin, other_sin = coeff[-3:]\\n    z = (x-offset)/scale\\n    for _ in range(50):\\n        w = z[:, other]\\n        r = offset + scale*z + coupling*w + own_sin*np.sin(freq*z) + other_sin*np.sin(freq*w) - x\\n        if np.max(abs(r), initial=0) < 1e-11:\\n            break\\n        jac = np.zeros((len(x), len(axis), len(axis)))\\n        jac[:, axis, axis] = scale + own_sin*freq*np.cos(freq*z)\\n        jac[:, axis, other] = coupling + other_sin*freq*np.cos(freq*w)\\n        step = np.linalg.solve(jac, r[..., None])[..., 0]\\n        z -= step / np.maximum(1., np.max(abs(step), axis=1, keepdims=True))\\n    return z\\n\\n\\ndef _product(z):\\n    return z[:,[0,0,1,2]]*z[:,[1,3,2,3]]\\n\\n\\ndef _cosine(z):\\n    return np.cos(2.4*z)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    s = inputs[\'__input__\']\\n    x = np.vstack((s[\'train\'][\'x\'], s[\'test\'][\'x\']))\\n    z = _calibration(s[\'panel_a\'], x)\\n    b, c = s[\'panel_b\'], s[\'panel_c\']\\n    p, r = _product(z), _cosine(z)\\n    return dict(z=z, L=_ones(z) @ _ls(_ones(b[\'u\']), b[\'v\']),\\n                Q=_ones(p) @ _ls(_ones(_product(c[\'u\'])), c[\'v1\']),\\n                R=_ones(r) @ _ls(_ones(_cosine(c[\'u\'])), c[\'v2\']),\\n                product=p, cosine=r)\\n", "node_01": "import numpy as np\\nfrom scipy.linalg import cho_factor, cho_solve\\n\\n\\ndef _ls(x, y):\\n    return np.linalg.lstsq(x, y, rcond=None)[0]\\n\\n\\ndef _ones(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _dist(a):\\n    return np.maximum(0, ((a[:, None, :] - a[None, :, :]) ** 2).sum(-1))\\n\\n\\ndef _models(z, l, q, r, p, c):\\n    cross = (p[:, :, None] * c[:, None, :]).reshape(len(z), -1)\\n    ds = _dist(z)\\n    fs = [_ones(np.c_[p, c, cross]), _ones(np.c_[z, p, c, cross]),\\n          _ones(np.c_[p, c, (q * r).sum(1)]), np.ones((len(z), 1))]\\n    for ix, F in enumerate(fs):\\n        lin = F @ F.T / F.shape[1]\\n        for D in [ds, _dist(np.c_[q, r]), ds + .5 * _dist(np.c_[q, r]), ds + .5 * _dist(l)]:\\n            for h in [.9, 1.7, 3., 5.]:\\n                for K in [np.exp(-D / h**2), (1 + D / h**2)**(-2)]:\\n                    for w in [.1, 1., 10., 100.] if ix < 3 else [1.]:\\n                        yield K + w * lin\\n    yield fs[0] @ fs[0].T / fs[0].shape[1]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    t = inputs[\'__input__\'][\'train\']\\n    features = inputs[\'node_00\']\\n    z, l, q, r, product, cosine = (features[k] for k in (\'z\', \'L\', \'Q\', \'R\', \'product\', \'cosine\'))\\n    n = len(t[\'x\'])\\n    traj = t[\'traj\']\\n    _, _, V = np.linalg.svd(traj, full_matrices=False)\\n    H = V[:3]\\n    target = np.c_[t[\'y\'], traj @ H.T]\\n    count = target.shape[1]\\n    models = [[] for _ in range(count)]\\n    I = np.eye(n)\\n    for K in _models(z, l, q, r, product, cosine):\\n        for lam in [1e-6, 1e-4, .002, .025]:\\n            f = cho_factor(K[:n, :n] + lam * I, check_finite=False)\\n            alpha = cho_solve(f, target, check_finite=False)\\n            inv = cho_solve(f, I, check_finite=False).diagonal()\\n            e = np.mean(abs(alpha / inv[:, None]), axis=0)\\n            pred = K[n:, :n] @ alpha\\n            for j in range(count):\\n                models[j].append((e[j], pred[:, j]))\\n    out = []\\n    for m in models:\\n        best = sorted(m, key=lambda v: v[0])[:10]\\n        w = np.exp(-15 * (np.array([a for a, b in best]) / best[0][0] - 1))\\n        out.append(np.average(np.stack([b for a, b in best]), axis=0, weights=w))\\n    pred = np.array(out).T\\n    bilinear = (product[:, :, None] * cosine[:, None, :]).reshape(len(product), -1)\\n    F = _ones(np.c_[product, cosine, bilinear])\\n    Y = np.c_[t[\'y\'], traj]\\n    E = Y - F[:n] @ _ls(F[:n], Y)\\n    _, sig, unit = np.linalg.svd(E, full_matrices=False)\\n    load = unit[0]\\n    if abs(load[0]) > .15 and sig[0] > 5 * sig[1]:\\n        ratio = load[1:] / load[0]\\n        orthogonal = traj - t[\'y\'][:, None] * ratio\\n        state = F[n:] @ _ls(F[:n], orthogonal) + pred[:, 0, None] * ratio\\n    else:\\n        state = pred[:, 1:] @ H\\n    y = pred[:, 0]\\n    end = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state, \'diagnostics_vec\': np.array([y.mean(), y.std(), end.mean(), end.std()])}\\n"}'

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
