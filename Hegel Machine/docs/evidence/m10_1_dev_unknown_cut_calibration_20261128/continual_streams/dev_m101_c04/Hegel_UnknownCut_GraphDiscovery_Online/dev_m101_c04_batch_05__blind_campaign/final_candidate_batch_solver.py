"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_04_r02", "depth": 3, "fanout_by_node": {"node_00": 1, "node_01": 0, "node_02": 1}, "incoming": {"node_00": ["node_02"], "node_01": ["__input__", "node_00"], "node_02": ["__input__"]}, "max_fanout": 1, "node_code_shas": {"node_00": "13be80cf69b567837d8d4f540c4dd8a57f142f7d888bf2d7ae3f3e0042f67190", "node_01": "b4ae5e17ed6f302736235e163fdd1fb9f48267e1cf0823fdab7b075f90cfd132", "node_02": "a3236ed8e0d09b2dd9fc97eb4bc43a6fc3addce297907dea484692c1b8927ecb"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_02", "node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _design(u, scales):\\n    x = u[:, :2]\\n    return np.column_stack((np.ones(len(u)), u,\\n                            x * np.exp(-scales[:2] * x**2),\\n                            np.sin(scales[2:] * u[:, 2:])))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    upstream = inputs[\'node_02\']\\n    u, v = upstream[\'b_u\'], upstream[\'b_v\']\\n\\n    def fit(scales):\\n        design = _design(u, scales)\\n        return design, np.linalg.lstsq(design, v, rcond=None)[0]\\n\\n    def residual(scales):\\n        design, coef = fit(scales)\\n        return (design @ coef - v).ravel()\\n\\n    scales = least_squares(residual, [1., 1., 3., 3.],\\n                           bounds=([.1, .1, 1., 1.], [5., 5., 6., 6.])).x\\n    _, coef = fit(scales)\\n    return {\'z_tr\': _design(upstream[\'u_tr\'], scales) @ coef,\\n            \'z_te\': _design(upstream[\'u_te\'], scales) @ coef}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _scalar_design(z):\\n    a, b, c, d = z.T\\n    return np.column_stack((np.ones(len(z)), z, a*c + b*d))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    z_train = np.asarray(inputs[\'node_00\'][\'z_tr\'], dtype=float)\\n    z_test = np.asarray(inputs[\'node_00\'][\'z_te\'], dtype=float)\\n    train = study[\'train\']\\n    x = _scalar_design(z_train)\\n    xt = _scalar_design(z_test)\\n    y = np.asarray(train[\'y\'], dtype=float)\\n    initial = np.linalg.lstsq(x, y, rcond=None)[0]\\n\\n    def predict(design, p):\\n        q = design @ p[:-2]\\n        return q + p[-2] * np.tanh(p[-1] * q)\\n\\n    p = least_squares(\\n        lambda p: predict(x, p) - y,\\n        np.r_[initial, 0.4, 1.7],\\n        bounds=(np.r_[np.full(x.shape[1], -np.inf), 0., 0.], np.inf),\\n    ).x\\n    y_hat = predict(xt, p)\\n\\n    weights = np.linalg.lstsq(x[:, :5], train[\'traj\'], rcond=None)[0]\\n    state_pred = xt[:, :5] @ weights\\n    terminal = state_pred[:, -1]\\n    return {\\n        \'y_hat\': y_hat,\\n        \'state_pred\': state_pred,\\n        \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()]),\\n    }\\n", "node_02": "import numpy as np\\n\\n\\ndef _fit_forward(u, v):\\n    neighbor = np.roll(u, -1, axis=1)\\n    design = np.stack((np.ones_like(u), u, neighbor, u * neighbor, u**3), axis=-1)\\n\\n    def fit(weights):\\n        return np.array([\\n            np.linalg.lstsq(design[:, j] * weights[:, None], v[:, j] * weights, rcond=None)[0]\\n            for j in range(u.shape[1])\\n        ]).T\\n\\n    coef = fit(np.ones(len(u)))\\n    for _ in range(30):\\n        residual = np.linalg.norm(v - np.einsum(\'njk,kj->nj\', design, coef), axis=1)\\n        scale = max(4e-4, 1.5 * np.median(residual))\\n        coef = fit(1.0 / np.hypot(1.0, residual / scale))\\n    residual = np.linalg.norm(v - np.einsum(\'njk,kj->nj\', design, coef), axis=1)\\n    keep = np.argsort(residual)[:-3]\\n    weights = np.zeros(len(u))\\n    weights[keep] = 1\\n    return fit(weights)\\n\\n\\ndef _invert(v, coef):\\n    bias, linear, coupling, interaction, cubic = coef\\n    j = np.arange(len(bias))\\n    k = (j + 1) % len(bias)\\n    affine = np.diag(linear)\\n    affine[j, k] = coupling\\n    u = np.linalg.solve(affine, (v - bias).T).T.copy()\\n    jacobian = np.broadcast_to(affine, (len(u), *affine.shape)).copy()\\n    for _ in range(25):\\n        neighbor = u[:, k]\\n        residual = bias + linear * u + coupling * neighbor + interaction * u * neighbor + cubic * u**3 - v\\n        if np.max(np.abs(residual)) < 1e-11:\\n            break\\n        jacobian[:, j, j] = linear + interaction * neighbor + 3 * cubic * u**2\\n        jacobian[:, j, k] = coupling + interaction * u\\n        u -= np.linalg.solve(jacobian, residual[..., None])[..., 0]\\n    return u\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    a, b = study[\'panel_a\'], study[\'panel_b\']\\n    coef = _fit_forward(a[\'u\'], a[\'v\'])\\n    return {\'u_tr\': _invert(study[\'train\'][\'x\'], coef),\\n            \'u_te\': _invert(study[\'test\'][\'x\'], coef),\\n            \'b_u\': b[\'u\'], \'b_v\': b[\'v\']}\\n"}'

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
