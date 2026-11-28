"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_07", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"], "node_02": ["node_00", "node_01"]}, "max_fanout": 2, "node_code_shas": {"node_00": "7fd74cc59cde6afe004242280a590aee6842a152756de567d205de932ffd42a5", "node_01": "5bb73605a2c70d5fa9c05e325bf5d41b94944696bc6cf4312c9af81b730140e6", "node_02": "2ed0920c3f195b8425360ba5d8c296315fe13639241967868f1032ebd2933861"}, "shared_fanout_nodes": ["node_00"], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _fit(x, y):\\n    return np.linalg.lstsq(x, y, rcond=None)[0]\\n\\n\\ndef _calibration(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    i, j = np.triu_indices(len(u), k=1)\\n    du = u[j] - u[i]\\n    dv = v[j] - v[i]\\n    gain = np.array([np.median(dv[du[:, k] != 0, k] / du[du[:, k] != 0, k])\\n                     for k in range(u.shape[1])])\\n    bias = np.median(v - u * gain, axis=0)\\n    error = np.linalg.norm(v - u * gain - bias, axis=1)\\n    center = np.median(error)\\n    inlier = error <= center + 8 * np.median(np.abs(error - center))\\n    bias, gain = np.array([_fit(_affine(u[inlier, k:k+1]), v[inlier, k])\\n                           for k in range(u.shape[1])]).T\\n    return bias, gain\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    bias, gain = _calibration(study[\'panel_a\'])\\n    return {\\n        \'x_tr_cal\': (study[\'train\'][\'x\'] - bias) / gain,\\n        \'x_te_cal\': (study[\'test\'][\'x\'] - bias) / gain,\\n    }\\n", "node_01": "import numpy as np\\n\\n\\ndef _basis(x, frequency, amplitude):\\n    return np.column_stack((np.ones(len(x)), x + amplitude * np.sin(frequency * x)))\\n\\n\\ndef _fit_panel(u, v):\\n    # Profile out the affine mixing for each shared coordinate warp.\\n    def evaluate(frequency, amplitude):\\n        design = _basis(u, frequency, amplitude)\\n        weights = np.linalg.lstsq(design, v, rcond=None)[0]\\n        error = np.sum((design @ weights - v) ** 2)\\n        return error, frequency, amplitude, weights\\n\\n    best = min((evaluate(f, a) for f in np.linspace(1.5, 4.5, 13)\\n                for a in np.linspace(-0.8, 0.0, 9)), key=lambda item: item[0])\\n    df, da = 0.125, 0.05\\n    for _ in range(48):\\n        _, f, a, _ = best\\n        trial = min((evaluate(f + i * df, a + j * da)\\n                     for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j),\\n                    key=lambda item: item[0])\\n        if trial[0] < best[0]:\\n            best = trial\\n        else:\\n            df *= 0.5\\n            da *= 0.5\\n        if df < 1e-6:\\n            break\\n    return best[1:]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    calibrated = inputs[\'node_00\']\\n    panel = study[\'panel_b\']\\n    frequency, amplitude, weights = _fit_panel(panel[\'u\'], panel[\'v\'])\\n    return {\\n        \'z_tr\': _basis(calibrated[\'x_tr_cal\'], frequency, amplitude) @ weights,\\n        \'z_te\': _basis(calibrated[\'x_te_cal\'], frequency, amplitude) @ weights,\\n        \'y\': study[\'train\'][\'y\'],\\n        \'traj\': study[\'train\'][\'traj\'],\\n    }\\n", "node_02": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _warp(x, params):\\n    frequency, amplitude = params\\n    return x + amplitude * np.sin(frequency * x)\\n\\n\\ndef _profile(x, target, start):\\n    def residual(params):\\n        design = _affine(_warp(x, params))\\n        weights = np.linalg.lstsq(design, target, rcond=None)[0]\\n        return (design @ weights - target).ravel()\\n\\n    return least_squares(residual, start, bounds=([1.5, -.8], [4.5, 0.])).x\\n\\n\\ndef _scalar_design(x, z, modulation):\\n    return np.column_stack((_affine(z), z[:, 0] * z[:, 2], z[:, 1] * z[:, 3],\\n                            z[:, 0] * modulation ** 2))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    c, o = inputs[\'node_00\'], inputs[\'node_01\']\\n    x, xt = c[\'x_tr_cal\'], c[\'x_te_cal\']\\n    z, zt, y, traj = o[\'z_tr\'], o[\'z_te\'], o[\'y\'], o[\'traj\']\\n\\n    # Recover the incoming operator\'s affine mixing without accessing its calibration panel.\\n    incoming = _profile(x, z, (3., -.4))\\n    mixing = np.linalg.lstsq(_affine(_warp(x, incoming)), z, rcond=None)[0][1:]\\n    # Trajectory labels identify the shared coordinate correction on training rows.\\n    corrected = _profile(x, traj, incoming)\\n    u, ut = _warp(x, corrected), _warp(xt, corrected)\\n    z = z + (u - _warp(x, incoming)) @ mixing\\n    zt = zt + (ut - _warp(xt, incoming)) @ mixing\\n\\n    base = _scalar_design(x, z, u[:, 1])\\n    query = _scalar_design(xt, zt, ut[:, 1])\\n\\n    def contrast(coords, params):\\n        return np.sin(params[0] * coords[:, 1] - params[1] * coords[:, 3])\\n\\n    def residual(params):\\n        design = np.column_stack((base, contrast(x, params)))\\n        return design @ np.linalg.lstsq(design, y, rcond=None)[0] - y\\n\\n    fit = min((least_squares(residual, (f, f), bounds=([.4, .3], [5., 5.]))\\n               for f in (1.2, 2.2, 3.6)), key=lambda result: result.cost)\\n    design = np.column_stack((base, contrast(x, fit.x)))\\n    query = np.column_stack((query, contrast(xt, fit.x)))\\n    y_hat = query @ np.linalg.lstsq(design, y, rcond=None)[0]\\n    state_pred = _affine(ut) @ np.linalg.lstsq(_affine(u), traj, rcond=None)[0]\\n    terminal = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])}\\n"}'

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
