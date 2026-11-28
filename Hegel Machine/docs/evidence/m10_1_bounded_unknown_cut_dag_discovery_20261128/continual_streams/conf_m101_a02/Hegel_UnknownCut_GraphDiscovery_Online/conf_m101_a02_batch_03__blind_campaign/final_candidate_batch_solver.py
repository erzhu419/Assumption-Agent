"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_04", "depth": 3, "fanout_by_node": {"node_00": 1, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"], "node_02": ["__input__", "node_01"]}, "max_fanout": 1, "node_code_shas": {"node_00": "cea124f0bfd10dff48ffa46d85e053804229e279dfd483ebe47525c07b4acb19", "node_01": "301dd96f03a5082e3549d67f9c368cfa6bcb75389cb60ea47c62d29477a64ea2", "node_02": "48d204177042a65bf25958deb0648de733e6fedc748ecc5df4f84c4005fb2285"}, "shared_fanout_nodes": [], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _fit_forward(u, v):\\n    # A cyclic neighbour modulates each coordinate\'s gain.\\n    neighbour = np.roll(u, -1, axis=1)\\n    design = np.stack((np.ones_like(u), u, u**3, (u - 1) * neighbour), axis=-1)\\n\\n    def fit(weights):\\n        w = np.sqrt(weights)\\n        return np.stack([np.linalg.lstsq(design[j] * w[:, None], v[:, j] * w,\\n                                        rcond=None)[0] for j in range(u.shape[1])])\\n\\n    design = design.transpose(1, 0, 2)\\n    weights = np.ones(len(u))\\n    for _ in range(30):\\n        coeff = fit(weights)\\n        error = np.linalg.norm(v - np.einsum(\'jnp,jp->nj\', design, coeff), axis=1)\\n        scale = max(1.8 * np.median(error), 1e-8)\\n        weights = 1 / (1 + (error / scale)**2)\\n    # The calibration protocol contains three corrupted rows.\\n    weights = np.ones(len(u))\\n    weights[np.argsort(error)[-3:]] = 0\\n    return fit(weights)\\n\\n\\ndef _invert(y, coeff):\\n    offset, gain, cubic, cross = coeff.T\\n    dim = len(gain)\\n    rows = np.arange(dim)\\n    next_rows = (rows + 1) % dim\\n    linear = np.diag(gain)\\n    linear[rows, next_rows] = -cross\\n    x = np.linalg.solve(linear, (y - offset).T).T\\n\\n    def forward(z):\\n        return offset + gain * z + cubic * z**3 + cross * (z - 1) * np.roll(z, -1, axis=1)\\n\\n    for _ in range(40):\\n        r = forward(x) - y\\n        norm = np.linalg.norm(r, axis=1)\\n        if np.max(norm) < 1e-11:\\n            break\\n        jac = np.zeros((len(x), dim, dim))\\n        jac[:, rows, rows] = gain + 3 * cubic * x**2 + cross * np.roll(x, -1, axis=1)\\n        jac[:, rows, next_rows] = cross * (x - 1)\\n        delta = np.linalg.solve(jac, r[..., None])[..., 0]\\n        step = np.ones(len(x))\\n        for _ in range(18):\\n            trial = x - step[:, None] * delta\\n            improved = np.linalg.norm(forward(trial) - y, axis=1) < norm\\n            if np.all(improved | (norm < 1e-11)):\\n                break\\n            step[~improved] *= 0.5\\n        x[improved] = trial[improved]\\n    return x\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    panel = study[\'panel_a\']\\n    coeff = _fit_forward(panel[\'u\'], panel[\'v\'])\\n    train, test = study[\'train\'][\'x\'], study[\'test\'][\'x\']\\n    calibrated = _invert(np.concatenate((train, test), axis=0), coeff)\\n    return {\'cal_tr\': calibrated[:len(train)], \'cal_te\': calibrated[len(train):],\\n            \'panel_b\': study[\'panel_b\']}\\n", "node_01": "import numpy as np\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    calibrated = inputs[\'node_00\']\\n    panel = calibrated[\'panel_b\']\\n    mixer = np.linalg.lstsq(_affine(panel[\'u\']), panel[\'v\'], rcond=None)[0]\\n    return {\\n        \'z_tr\': _affine(calibrated[\'cal_tr\']) @ mixer,\\n        \'z_te\': _affine(calibrated[\'cal_te\']) @ mixer,\\n    }\\n", "node_02": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _fit(design, values):\\n    return np.linalg.lstsq(design, values, rcond=None)[0]\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _label_basis(z):\\n    paired_products = z[:, 0] * z[:, 2] + z[:, 1] * z[:, 3]\\n    return np.column_stack((_affine(z), paired_products))\\n\\n\\ndef _trajectory(design, observed, design_test, times):\\n    affine_coeff = _fit(design, observed)\\n\\n    def time_basis(log_rates):\\n        return -np.expm1(-np.exp(log_rates)[:, None] * times)\\n\\n    def profile(log_rates):\\n        basis = time_basis(log_rates)\\n        amplitudes = _fit(basis.T, affine_coeff.T).T\\n        return amplitudes @ basis\\n\\n    def residual(log_rates):\\n        return (design @ (affine_coeff - profile(log_rates))).ravel()\\n\\n    rates = least_squares(residual, np.log([0.4, 1.8]), bounds=(-4, 3)).x\\n    return design_test @ profile(rates)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    mixed = inputs[\'node_01\']\\n    z_train, z_test = mixed[\'z_tr\'], mixed[\'z_te\']\\n    y_hat = _label_basis(z_test) @ _fit(_label_basis(z_train), train[\'y\'])\\n    state_pred = _trajectory(_affine(z_train), train[\'traj\'], _affine(z_test),\\n                             study[\'eval_times\'])\\n    final = state_pred[:, -1]\\n    diagnostics = np.array([y_hat.mean(), y_hat.std(), final.mean(), final.std()])\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred, \'diagnostics_vec\': diagnostics}\\n"}'

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
