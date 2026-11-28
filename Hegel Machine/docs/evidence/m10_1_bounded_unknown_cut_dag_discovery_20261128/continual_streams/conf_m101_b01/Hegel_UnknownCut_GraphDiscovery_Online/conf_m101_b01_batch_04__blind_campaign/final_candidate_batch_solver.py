"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_05", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"], "node_02": ["__input__", "node_00", "node_01"]}, "max_fanout": 2, "node_code_shas": {"node_00": "ccf5a60f66913ccae85e962a2bed270be9ecb1451736cf5547ae9db21b03f6c0", "node_01": "918cc73ffe65c63a713176af532c636e67b2f56dc7b672f93544009d25c19cb3", "node_02": "7ae82dee14f687fdaf2dabecfa0ac449c1f428627a3605e25a955fa13cb09cf3"}, "shared_fanout_nodes": ["node_00"], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _calibration(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    rng = np.random.default_rng(452)\\n    best = (np.inf, None, None)\\n    for _ in range(100):\\n        i, j = rng.choice(len(u), 2, replace=False)\\n        d = u[j] - u[i]\\n        if np.min(np.abs(d)) < 0.06:\\n            continue\\n        gain = (v[j] - v[i]) / d\\n        offset = v[i] - gain * u[i]\\n        err = np.linalg.norm(u * gain + offset - v, axis=1)\\n        score = np.median(err)\\n        if score < best[0]:\\n            best = (score, gain, offset)\\n    _, gain, offset = best\\n    err = np.linalg.norm(u * gain + offset - v, axis=1)\\n    keep = err <= max(0.001, 5 * np.median(err))\\n    uc, vc = u[keep], v[keep]\\n    a = uc - uc.mean(axis=0)\\n    gain = np.sum(a * (vc - vc.mean(axis=0)), axis=0) / np.sum(a * a, axis=0)\\n    return gain, vc.mean(axis=0) - uc.mean(axis=0) * gain\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    x = np.concatenate((study[\'train\'][\'x\'], study[\'test\'][\'x\']))\\n    gain, offset = _calibration(study[\'panel_a\'])\\n    return {\'calibrated_x\': (x - offset) / gain, \'panel_b\': study[\'panel_b\']}\\n", "node_01": "import numpy as np\\n\\n\\ndef _affine_features(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    upstream = inputs[\'node_00\']\\n    panel = upstream[\'panel_b\']\\n    operator = np.linalg.lstsq(_affine_features(panel[\'u\']), panel[\'v\'], rcond=None)[0]\\n    return {\'x\': _affine_features(upstream[\'calibrated_x\']) @ operator}\\n", "node_02": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _quadratic(x):\\n    i, j = np.triu_indices(x.shape[1])\\n    return x[:, i] * x[:, j]\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _scalar(u, z, y):\\n    # Calibrated-by-transformed terms use both frozen upstream outputs.\\n    h = np.tanh(2.0 * z)\\n    D = _affine(np.column_stack((z, h, _quadratic(h), u * h)))\\n    n = len(y)\\n    scale = np.maximum(np.sqrt(np.mean(D[:n] ** 2, axis=0)), 1e-9)\\n    D = D / scale\\n    Q, s, V = np.linalg.svd(D[:n], full_matrices=False)\\n    keep = s > 1e-7\\n    Q = Q[:, keep]\\n    weights = V[keep].T / s[keep]\\n    transfer = D[n:] @ weights\\n    ridge = 0.16 * weights[1:]\\n    penalty_jac = np.column_stack((ridge, np.zeros((len(ridge), 2))))\\n    power = 0.55\\n\\n    def link(x):\\n        return x / (1.0 + np.abs(x) ** (2 * power)) ** (0.5 / power)\\n\\n    def residual(a):\\n        b, amplitude, offset = a[:-2], a[-2], a[-1]\\n        return np.r_[amplitude * link(Q @ b) + offset - y, ridge @ b]\\n\\n    def jacobian(a):\\n        x = Q @ a[:-2]\\n        derivative = (1.0 + np.abs(x) ** (2 * power)) ** (-1.0 - 0.5 / power)\\n        data_jac = np.column_stack((a[-2] * derivative[:, None] * Q,\\n                                    link(x), np.ones(n)))\\n        return np.vstack((data_jac, penalty_jac))\\n\\n    initial = Q.T @ y\\n    fits = [least_squares(residual, np.r_[initial / m, m, y.mean()],\\n                           jac=jacobian, max_nfev=60)\\n            for m in (0.5, 1.0, 2.0)]\\n    a = min(fits, key=lambda fit: fit.cost).x\\n    return a[-2] * link(transfer @ a[:-2]) + a[-1]\\n\\n\\ndef _trajectory(z, panel, traj):\\n    q = panel[\'u\']\\n    v = panel[\'v1\']\\n\\n    def residual(f):\\n        A = _affine(np.column_stack((q, np.tanh(f * q))))\\n        return (A @ np.linalg.lstsq(A, v, rcond=None)[0] - v).ravel()\\n\\n    f = least_squares(residual, np.full(q.shape[1], 2.2), bounds=(.15, 8)).x\\n    h = np.tanh(z * f)\\n    D = np.column_stack((z, h, np.tanh(h), np.sin(h)))\\n    n = len(traj)\\n    center = D[:n].mean(axis=0)\\n    scale = np.maximum(D[:n].std(axis=0), 1e-10)\\n    D = (D - center) / scale\\n    U, s, V = np.linalg.svd(D[:n], full_matrices=False)\\n    avg = traj.mean(axis=0)\\n    coef = (V.T * (s / (s ** 2 + .0001))) @ (U.T @ (traj - avg))\\n    return avg + D[n:] @ coef\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    u = inputs[\'node_00\'][\'calibrated_x\']\\n    z = inputs[\'node_01\'][\'x\']\\n    train = study[\'train\']\\n    y = _scalar(u, z, train[\'y\'])\\n    state = _trajectory(z, study[\'panel_c\'], train[\'traj\'])\\n    tail = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), tail.mean(), tail.std()])}\\n"}'

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
