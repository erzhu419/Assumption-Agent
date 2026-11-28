"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_04", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "90580cb32536c76052872d342528d674a561eef7798f676a5f7b4b1cf61b4682", "node_01": "04cca5c9e6c451949c8d9d1b2d3a5fe8bb0015b8b227f37451abcea916d72c5f"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import root\\n\\n\\ndef _affine(u):\\n    return np.column_stack((np.ones(len(u)), u))\\n\\n\\ndef _robust_fit(design, values, weights):\\n    for _ in range(30):\\n        coef = np.linalg.lstsq(\\n            (design * weights[:, None, None]).reshape(-1, design.shape[-1]),\\n            (values * weights[:, None]).ravel(), rcond=None,\\n        )[0]\\n        residual = np.linalg.norm(design @ coef - values, axis=1)\\n        weights = 1 / np.hypot(1, residual / max(1e-6, np.median(residual)))\\n    return coef, residual, weights\\n\\n\\ndef _fit_panel_a(u, v):\\n    n, d = u.shape\\n    eye = np.eye(d)\\n    linear = (_affine(u)[:, None, :, None] * eye[None, :, None, :]).reshape(n, d, -1)\\n    _, _, weights = _robust_fit(linear, v, np.ones(n))\\n\\n    # Each channel couples to its cyclic successor with shared linear,\\n    # quadratic and cubic gains; offsets and diagonal slopes are per channel.\\n    next_u = np.roll(u, -1, axis=1)\\n    design = np.concatenate((\\n        np.broadcast_to(eye, (n, d, d)), eye[None, :, :] * u[:, :, None],\\n        np.stack((next_u, u * next_u, u ** 3), axis=-1),\\n    ), axis=-1)\\n    _, residual, _ = _robust_fit(design, v, weights)\\n    good = np.argsort(residual)[:-3]\\n    return np.linalg.lstsq(design[good].reshape(-1, 2*d+3), v[good].ravel(), rcond=None)[0]\\n\\n\\ndef _invert_panel_a(x, coef):\\n    d = x.shape[1]\\n    j = np.arange(d)\\n    nxt = (j + 1) % d\\n    bias, slope = coef[:d], coef[d:2*d]\\n    coupling, pair, cubic = coef[-3:]\\n    initial_jac = np.diag(slope)\\n    initial_jac[j, nxt] = coupling\\n    initial = np.linalg.solve(initial_jac, (x - bias).T).T\\n\\n    def residual(u, observation):\\n        return bias + slope*u + (coupling + pair*u)*u[nxt] + cubic*u**3 - observation\\n\\n    def jacobian(u, observation):\\n        jac = np.diag(slope + pair*u[nxt] + 3*cubic*u*u)\\n        jac[j, nxt] = coupling + pair*u\\n        return jac\\n\\n    return np.array([\\n        root(residual, guess, args=(row,), jac=jacobian, options={\'xtol\': 1e-10}).x\\n        for row, guess in zip(x, initial)\\n    ]).reshape(x.shape)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    a, b = study[\'panel_a\'], study[\'panel_b\']\\n    coef_a = _fit_panel_a(a[\'u\'], a[\'v\'])\\n    coef_b = np.linalg.lstsq(_affine(b[\'u\']), b[\'v\'], rcond=None)[0]\\n    return {\\n        key: _affine(_invert_panel_a(study[split][\'x\'], coef_a)) @ coef_b\\n        for key, split in ((\'z_tr\', \'train\'), (\'z_te\', \'test\'))\\n    }\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _fit(design, values):\\n    return np.linalg.lstsq(design, values, rcond=None)[0]\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _label_basis(z):\\n    paired_products = z[:, 0] * z[:, 2] + z[:, 1] * z[:, 3]\\n    return np.column_stack((_affine(z), paired_products))\\n\\n\\ndef _trajectory(design, observed, design_test, times):\\n    affine_coeff = _fit(design, observed)\\n\\n    def time_basis(log_rates):\\n        return -np.expm1(-np.exp(log_rates)[:, None] * times)\\n\\n    def profile(log_rates):\\n        basis = time_basis(log_rates)\\n        amplitudes = _fit(basis.T, affine_coeff.T).T\\n        return amplitudes @ basis\\n\\n    def residual(log_rates):\\n        return (design @ (affine_coeff - profile(log_rates))).ravel()\\n\\n    rates = least_squares(residual, np.log([0.4, 1.8]), bounds=(-4, 3)).x\\n    return design_test @ profile(rates)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    z_train = inputs[\'node_00\'][\'z_tr\']\\n    z_test = inputs[\'node_00\'][\'z_te\']\\n    y_hat = _label_basis(z_test) @ _fit(_label_basis(z_train), train[\'y\'])\\n    state_pred = _trajectory(_affine(z_train), train[\'traj\'], _affine(z_test),\\n                             study[\'eval_times\'])\\n    final = state_pred[:, -1]\\n    diagnostics = np.array([y_hat.mean(), y_hat.std(), final.mean(), final.std()])\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred, \'diagnostics_vec\': diagnostics}\\n"}'

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
