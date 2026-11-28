"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_04", "depth": 3, "fanout_by_node": {"node_00": 1, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"], "node_02": ["__input__", "node_01"]}, "max_fanout": 1, "node_code_shas": {"node_00": "40f2fd6b04bd18625f4b7b11c8b602aedc2abc9eaf49cf37efa0442d49585f70", "node_01": "8e24ebf14ed0da778d1e99da53e9a706bc78a63ef54f225bfa17fff5c1748c39", "node_02": "69ed01e55e97f9b44ba07a7175f340a09e1a53faf8e271127afa87baf1c26e7e"}, "shared_fanout_nodes": [], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _fit_forward(u, v):\\n    n, d = u.shape\\n    nxt = np.roll(u, -1, axis=1)\\n    axis = np.eye(d)\\n    # Axis-specific offsets and gains; the three nonlinear/coupled terms\\n    # share coefficients around the coordinate cycle.\\n    design = np.concatenate((\\n        np.broadcast_to(axis, (n, d, d)),\\n        u[:, :, None] * axis,\\n        np.stack((nxt, u * nxt, u**3), axis=-1),\\n    ), axis=-1)\\n    weight = np.ones(n)\\n    for _ in range(40):\\n        w = np.sqrt(weight)[:, None, None]\\n        coef = np.linalg.lstsq((design * w).reshape(n*d, -1),\\n                               (v * w[..., 0]).ravel(), rcond=None)[0]\\n        residual = np.linalg.norm(design @ coef - v, axis=1)\\n        scale = max(1e-10, 0.1 * np.median(residual))\\n        weight = scale / np.maximum(scale, residual)\\n    # Panel A contains three grossly corrupted observations.\\n    good = np.argsort(residual)[:-3]\\n    return np.linalg.lstsq(design[good].reshape(len(good)*d, -1),\\n                           v[good].ravel(), rcond=None)[0]\\n\\n\\ndef _invert(x, coef):\\n    d = x.shape[1]\\n    j = np.arange(d)\\n    k = (j + 1) % d\\n    bias, gain = coef[:d], coef[d:2*d]\\n    neighbor, pair, cubic = coef[2*d:]\\n\\n    def forward(u):\\n        return bias + gain*u + (neighbor + pair*u)*u[k] + cubic*u**3\\n\\n    def jacobian(u):\\n        jac = np.diag(gain + pair*u[k] + 3*cubic*u**2)\\n        jac[j, k] = neighbor + pair*u\\n        return jac\\n\\n    initial = np.linalg.solve(jacobian(np.zeros(d)), (x-bias).T).T\\n    result = np.empty_like(initial)\\n    for i, (row, start) in enumerate(zip(x, initial)):\\n        result[i] = least_squares(lambda u: forward(u) - row, start,\\n                                  jac=jacobian, gtol=1e-12).x\\n    return result\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    panel = study[\'panel_a\']\\n    coef = _fit_forward(panel[\'u\'], panel[\'v\'])\\n    n_train = len(study[\'train\'][\'x\'])\\n    calibrated = _invert(np.concatenate((study[\'train\'][\'x\'], study[\'test\'][\'x\'])), coef)\\n    return {\'u_tr\': calibrated[:n_train], \'u_te\': calibrated[n_train:],\\n            \'panel_b\': study[\'panel_b\']}\\n", "node_01": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    calibrated = inputs[\'node_00\']\\n    b = calibrated[\'panel_b\']\\n    design = np.column_stack((np.ones(len(b[\'u\'])), b[\'u\']))\\n    operator = np.linalg.lstsq(design, b[\'v\'], rcond=None)[0]\\n    return {\\n        \'z_tr\': calibrated[\'u_tr\'] @ operator[1:] + operator[0],\\n        \'z_te\': calibrated[\'u_te\'] @ operator[1:] + operator[0],\\n    }\\n", "node_02": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _fit(design, target):\\n    return np.linalg.lstsq(design, target, rcond=None)[0]\\n\\n\\ndef _scalar_design(z):\\n    interaction = z[:, 0] * z[:, 2] + z[:, 1] * z[:, 3]\\n    return np.column_stack((z, interaction))\\n\\n\\ndef _trajectory(x, target, x_test, t):\\n    def basis(rates):\\n        return -np.expm1(-t[:, None] * rates)\\n\\n    def residual(rates):\\n        b = basis(rates)\\n        return (b @ _fit(b, target.T) - target.T).ravel()\\n\\n    rates = least_squares(residual, [0.4, 1.8], bounds=([0.01, 0.8], [0.8, 8.0])).x\\n    b = basis(rates)\\n    return (x_test @ _fit(x, _fit(b, target.T).T)) @ b.T\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    mapped = inputs[\'node_01\']\\n    train = study[\'train\']\\n    x, x_test = mapped[\'z_tr\'], mapped[\'z_te\']\\n    y_hat = _scalar_design(x_test) @ _fit(_scalar_design(x), train[\'y\'])\\n    state_pred = _trajectory(x, train[\'traj\'], x_test, study[\'eval_times\'])\\n    end = state_pred[:, -1]\\n    diagnostics = np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred, \'diagnostics_vec\': diagnostics}\\n"}'

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
