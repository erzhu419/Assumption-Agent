"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_06", "depth": 3, "fanout_by_node": {"node_00": 1, "node_01": 0, "node_02": 2}, "incoming": {"node_00": ["node_02"], "node_01": ["__input__", "node_00", "node_02"], "node_02": ["__input__"]}, "max_fanout": 2, "node_code_shas": {"node_00": "9db6ba16dd6743363158c6bd381596ca9f06ed27f0911c7e3fc991c009999bac", "node_01": "38c4eebb08134e8fa58f4f4a05f352440918c242cb383594b2eec6aee6a69c41", "node_02": "a3236ed8e0d09b2dd9fc97eb4bc43a6fc3addce297907dea484692c1b8927ecb"}, "shared_fanout_nodes": ["node_02"], "sink_node_id": "node_01", "topo_order": ["node_02", "node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _affine(u):\\n    return np.column_stack((np.ones(len(u)), u))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    first = inputs[\'node_02\']\\n    coef = np.linalg.lstsq(_affine(first[\'b_u\']), first[\'b_v\'], rcond=None)[0]\\n    return {\'z_tr\': _affine(first[\'u_tr\']) @ coef,\\n            \'z_te\': _affine(first[\'u_te\']) @ coef}\\n", "node_01": "from itertools import combinations_with_replacement\\n\\nimport numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _quadratic(x):\\n    return np.column_stack((np.ones(len(x)), x, *(\\n        x[:, i] * x[:, j]\\n        for i, j in combinations_with_replacement(range(x.shape[1]), 2)\\n    )))\\n\\n\\ndef _scalar(q, z, y, q_test, z_test):\\n    # Profile out the quadratic and saturation amplitude while fitting the\\n    # direction and offset of a single saturation in the mixed coordinates.\\n    center, scale = z.mean(axis=0), z.std(axis=0)\\n    a = np.column_stack((np.ones(len(z)), (z - center) / scale))\\n    a_test = np.column_stack((np.ones(len(z_test)), (z_test - center) / scale))\\n    basis, _ = np.linalg.qr(q, mode=\'reduced\')\\n    residual = y - basis @ (basis.T @ y)\\n\\n    def objective(p):\\n        h = np.tanh(a @ p)\\n        h -= basis @ (basis.T @ h)\\n        return h * ((h @ residual) / max(h @ h, 1e-20)) - residual\\n\\n    rng = np.random.default_rng(442)\\n    starts = rng.normal(0.0, 0.7, (16, a.shape[1]))\\n    optimum = min((least_squares(objective, start, bounds=(-5.0, 5.0),\\n                                 max_nfev=160) for start in starts),\\n                  key=lambda fit: fit.cost).x\\n    features = np.column_stack((q, np.tanh(a @ optimum)))\\n    beta = np.linalg.lstsq(features, y, rcond=None)[0]\\n    return q_test @ beta[:-1] + beta[-1] * np.tanh(a_test @ optimum)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    train = inputs[\'__input__\'][\'train\']\\n    up, mixed = inputs[\'node_02\'], inputs[\'node_00\']\\n    q, q_test = _quadratic(up[\'u_tr\']), _quadratic(up[\'u_te\'])\\n    state = q_test @ np.linalg.lstsq(q, train[\'traj\'], rcond=None)[0]\\n    y = _scalar(q, mixed[\'z_tr\'], np.asarray(train[\'y\']), q_test, mixed[\'z_te\'])\\n    last = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), last.mean(), last.std()])}\\n", "node_02": "import numpy as np\\n\\n\\ndef _fit_forward(u, v):\\n    neighbor = np.roll(u, -1, axis=1)\\n    design = np.stack((np.ones_like(u), u, neighbor, u * neighbor, u**3), axis=-1)\\n\\n    def fit(weights):\\n        return np.array([\\n            np.linalg.lstsq(design[:, j] * weights[:, None], v[:, j] * weights, rcond=None)[0]\\n            for j in range(u.shape[1])\\n        ]).T\\n\\n    coef = fit(np.ones(len(u)))\\n    for _ in range(30):\\n        residual = np.linalg.norm(v - np.einsum(\'njk,kj->nj\', design, coef), axis=1)\\n        scale = max(4e-4, 1.5 * np.median(residual))\\n        coef = fit(1.0 / np.hypot(1.0, residual / scale))\\n    residual = np.linalg.norm(v - np.einsum(\'njk,kj->nj\', design, coef), axis=1)\\n    keep = np.argsort(residual)[:-3]\\n    weights = np.zeros(len(u))\\n    weights[keep] = 1\\n    return fit(weights)\\n\\n\\ndef _invert(v, coef):\\n    bias, linear, coupling, interaction, cubic = coef\\n    j = np.arange(len(bias))\\n    k = (j + 1) % len(bias)\\n    affine = np.diag(linear)\\n    affine[j, k] = coupling\\n    u = np.linalg.solve(affine, (v - bias).T).T.copy()\\n    jacobian = np.broadcast_to(affine, (len(u), *affine.shape)).copy()\\n    for _ in range(25):\\n        neighbor = u[:, k]\\n        residual = bias + linear * u + coupling * neighbor + interaction * u * neighbor + cubic * u**3 - v\\n        if np.max(np.abs(residual)) < 1e-11:\\n            break\\n        jacobian[:, j, j] = linear + interaction * neighbor + 3 * cubic * u**2\\n        jacobian[:, j, k] = coupling + interaction * u\\n        u -= np.linalg.solve(jacobian, residual[..., None])[..., 0]\\n    return u\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    a, b = study[\'panel_a\'], study[\'panel_b\']\\n    coef = _fit_forward(a[\'u\'], a[\'v\'])\\n    return {\'u_tr\': _invert(study[\'train\'][\'x\'], coef),\\n            \'u_te\': _invert(study[\'test\'][\'x\'], coef),\\n            \'b_u\': b[\'u\'], \'b_v\': b[\'v\']}\\n"}'

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
    outputs = {'__input__': deepcopy(study)}
    exec_counts = {nid: 0 for nid in _SPEC_OBJ['topo_order']}
    non_identity_checks = {}
    for nid in _SPEC_OBJ['topo_order']:
        srcs = _SPEC_OBJ['incoming'][nid]
        if len(_SPEC_OBJ['topo_order']) == 1 and srcs == ['__input__']:
            node_in = deepcopy(outputs['__input__'])
        else:
            node_in = {s: deepcopy(outputs[s]) for s in srcs}
        in_dig = _canon_digest(node_in)
        out_v = _COMPILED_FNS[nid](node_in)
        exec_counts[nid] += 1
        if exec_counts[nid] != 1:
            raise RuntimeError(f'DAG node {nid} executed {exec_counts[nid]} times != 1')
        if not isinstance(out_v, dict) or not out_v:
            raise RuntimeError(f'DAG node {nid} must return a non-empty dict')
        out_dig = _canon_digest(out_v)
        if nid != _SPEC_OBJ['sink_node_id']:
            # Anti-identity-shell check: intermediate node cannot merely echo its raw predecessor dict
            echoed = (out_dig == in_dig) or any(out_dig == _canon_digest(outputs[s]) for s in srcs)
            if echoed:
                raise RuntimeError(f'DAG node {nid} failed anti-identity-shell check')
            non_identity_checks[nid] = True
        outputs[nid] = out_v
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
            'node_code_shas': _SPEC_OBJ['node_code_shas'],
        }
        open(trace_fp, 'w', encoding='utf-8').write(json.dumps(trace_obj, sort_keys=True))
    return outputs[_SPEC_OBJ['sink_node_id']]
