"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_08", "depth": 3, "fanout_by_node": {"node_00": 1, "node_01": 2, "node_02": 1, "node_03": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__"], "node_02": ["node_00", "node_01"], "node_03": ["node_01", "node_02"]}, "max_fanout": 2, "node_code_shas": {"node_00": "73c6e1f6870ef0ae07c389c6ee502f68107ba7e759a03dcfe63bbb29d2ea424a", "node_01": "493aca178d528c16ec681d5e109ba4e4500ff58bbef4f1e9c17dd6aa7ee50743", "node_02": "9d9cf57d412595efc361c244c11a554b623747129b330a44ce7601a11d8fcffd", "node_03": "eeeaeecd9a19626365176f029b400d14a57213e504550a9bef596c3f286941ff"}, "shared_fanout_nodes": ["node_01"], "sink_node_id": "node_03", "topo_order": ["node_00", "node_01", "node_02", "node_03"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import least_squares, root\\n\\ndef lin(f, y):\\n    return np.linalg.lstsq(f, y, rcond=1e-9)[0]\\n\\ndef one(z):\\n    return np.c_[np.ones(len(z)), z]\\n\\ndef _panel_maps(panel, z):\\n    u = np.asarray(panel[\'u\'], dtype=float)\\n\\n    # Both responses share their nonlinear shape parameters across coordinates.\\n    def softsign(w, q):\\n        return one(w / (1 + q[0]*abs(w)) + q[1]*w)\\n\\n    def curvature(w, q):\\n        w2 = w*w\\n        return np.c_[one(w), w2, w2/(1+q[0]*w2)]\\n\\n    def fit(features, target, initial, bounds, **options):\\n        def residual(q):\\n            f = features(u, q)\\n            return (f @ lin(f, target) - target).ravel()\\n        q = least_squares(residual, initial, bounds=bounds, **options).x\\n        return features(z, q) @ lin(features(u, q), target)\\n\\n    c = fit(softsign, panel[\'v1\'], [.85, -.01], ([.02, -1], [10, 1]))\\n    d = fit(curvature, panel[\'v2\'], [.25], (.08, .65), gtol=1e-13)\\n    return np.c_[c, d]\\n\\n\\ndef _invert_panel(panel, x):\\n    u, v = (np.asarray(panel[k], dtype=float) for k in (\'u\', \'v\'))\\n    affine = one(u)\\n    keep = np.ones(len(u), dtype=bool)\\n    for _ in range(4):\\n        coef = lin(affine[keep], v[keep])\\n        keep = np.linalg.norm(v-affine@coef, axis=1) < 2\\n    nxt = np.roll(u, -1, axis=1)\\n    coef = np.column_stack([\\n        lin(np.column_stack((np.ones(len(u)), u[:, j], nxt[:, j],\\n                            u[:, j]*nxt[:, j], u[:, j]**3))[keep], v[keep, j])\\n        for j in range(u.shape[1])\\n    ])\\n    b, scale, mix, cross, cubic = coef\\n    aff = lin(affine[keep], v[keep])\\n    def forward(z):\\n        nz = np.roll(z, -1)\\n        return b + scale*z + mix*nz + cross*z*nz + cubic*z**3\\n    def jac(z):\\n        nz = np.roll(z, -1)\\n        return np.diag(scale + cross*nz + 3*cubic*z*z) + np.diag(mix+cross*z)@np.roll(np.eye(len(z)), -1, axis=0)\\n    init = np.linalg.solve(aff[1:].T, (x-aff[0]).T).T\\n    out = []\\n    for row, guess in zip(x, init):\\n        r = root(lambda z:forward(z)-row, guess, jac=jac, tol=1e-10)\\n        out.append(r.x if np.linalg.norm(r.fun) < 1e-8 else least_squares(lambda z:forward(z)-row, r.x, jac=jac).x)\\n    return np.asarray(out)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train, test = study[\'train\'][\'x\'], study[\'test\'][\'x\']\\n    n = len(train)\\n    z = _invert_panel(study[\'panel_a\'], np.concatenate((train, test)))\\n    panel = study[\'panel_c\']\\n    u, v = panel[\'u\'], panel[\'v1\']\\n    def design(x, q):\\n        return one(np.column_stack((x, np.tanh(q[0]*x), x*np.abs(x))))\\n    def residual(q):\\n        f = design(u, q)\\n        return (f@lin(f, v)-v).ravel()\\n    q = min((least_squares(residual, [s], bounds=(0.02, 10), max_nfev=300)\\n             for s in (0.25, 0.85, 1.6, 2.5)), key=lambda fit:fit.cost).x\\n    c = design(z, q)@lin(design(u, q), v)\\n    d = _panel_maps(panel, z)[:, u.shape[1]:]\\n    w = np.column_stack((c, d))\\n    w = (w-w[:n].mean(axis=0))/np.maximum(w[:n].std(axis=0), 0.1)\\n    return {\'features\': w, \'n_train\': np.array(n)}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef lin(f, y):\\n    return np.linalg.lstsq(f, y, rcond=1e-9)[0]\\n\\n\\ndef _time_basis(t, trajectory):\\n    def basis(r):\\n        return -np.expm1(-t[:, None]*r)\\n\\n    def residual(r):\\n        f = basis(r)\\n        return (f @ lin(f, trajectory.T) - trajectory.T).ravel()\\n\\n    rates = least_squares(residual, [.4, 1.8], bounds=([.05, .8], [.8, 4])).x\\n    return basis(rates)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    targets = np.column_stack((train[\'y\'], train[\'traj\']))\\n    basis = _time_basis(np.asarray(study[\'eval_times\']), train[\'traj\'])\\n    return {\'targets\': targets, \'basis\': basis}\\n", "node_02": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    feature_data = inputs[\'node_00\']\\n    w = feature_data[\'features\']\\n    n = feature_data[\'n_train\']\\n    targets = inputs[\'node_01\'][\'targets\']\\n    gram = w @ w.T / w.shape[1]\\n    norms = gram.diagonal()\\n    distance = np.maximum(norms[:, None] + norms - 2*gram, 0)\\n    quadratic = (1+.25*gram)**2\\n    eye = np.eye(n)\\n    fit = np.mean([\\n        kernel[n:, :n] @ np.linalg.solve(kernel[:n, :n] + ridge*eye, targets)\\n        for kernel, ridge in ((quadratic, 3e-5),\\n                              (quadratic + np.exp(-.06*distance), 3e-4))\\n    ], axis=0)\\n    return {\'fit\': fit}\\n", "node_03": "import numpy as np\\n\\n\\ndef lin(f, y):\\n    return np.linalg.lstsq(f, y, rcond=1e-9)[0]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    basis = inputs[\'node_01\'][\'basis\']\\n    fit = inputs[\'node_02\'][\'fit\']\\n    y = fit[:, 0]\\n    state = (basis @ lin(basis, fit[:, 1:].T)).T\\n    terminal = state[:, -1]\\n    diagnostics = np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])\\n    return {\'y_hat\': y, \'state_pred\': state, \'diagnostics_vec\': diagnostics}\\n"}'

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
