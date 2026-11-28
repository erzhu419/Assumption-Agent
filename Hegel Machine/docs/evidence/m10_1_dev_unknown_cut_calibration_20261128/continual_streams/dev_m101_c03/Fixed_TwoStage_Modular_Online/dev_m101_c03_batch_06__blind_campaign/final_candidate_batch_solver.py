"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_07", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "75e05742ab5d0f6f4ab59b391c13235bcd068370a9445e6925ab30654fd00f06", "node_01": "58e6c561f5404291c1a7591c8a1a0758cdfae9f359cce935f0d7bf443adbe5e7"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import least_squares, minimize_scalar\\n\\n\\ndef _intercept(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _fit(x, y):\\n    return np.linalg.lstsq(x, y, rcond=None)[0]\\n\\n\\ndef _panel_a_inverse(panel, observed):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    i = np.arange(u.shape[1])\\n    j = (i + 1) % len(i)\\n    next_u = u[:, j]\\n    design = np.stack((np.ones_like(u), u, next_u, u * next_u, u**3), axis=2)\\n    weights = np.ones(len(u))\\n    for _ in range(25):\\n        coef = np.column_stack([_fit(design[:, k] * weights[:, None], v[:, k] * weights) for k in i])\\n        errors = np.linalg.norm(np.einsum(\'nik,ki->ni\', design, coef) - v, axis=1)\\n        scale = max(1e-6, 1.5 * np.median(errors))\\n        weights = 1 / np.sqrt(1 + (errors / scale)**2)\\n\\n    offset, gain, coupling, cross, cubic = coef\\n\\n    def forward(z):\\n        return offset + gain * z + coupling * z[j] + cross * z * z[j] + cubic * z**3\\n\\n    def jacobian(z):\\n        jac = np.diag(gain + cross * z[j] + 3 * cubic * z**2)\\n        jac[i, j] += coupling + cross * z\\n        return jac\\n\\n    return np.array([least_squares(lambda z: forward(z) - row, row - offset,\\n                                  jac=jacobian, gtol=1e-11).x for row in observed])\\n\\n\\ndef _panel_c_features(panel, u):\\n    def pair_design(z):\\n        return _intercept(z * np.roll(z, -1, axis=1))\\n\\n    def cosine_design(z, frequency):\\n        return _intercept(np.cos(frequency * z))\\n\\n    ref = panel[\'u\']\\n    c = pair_design(u) @ _fit(pair_design(ref), panel[\'v1\'])\\n\\n    def cosine_loss(frequency):\\n        a = cosine_design(ref, frequency)\\n        return np.sum((a @ _fit(a, panel[\'v2\']) - panel[\'v2\'])**2)\\n\\n    frequency = minimize_scalar(cosine_loss, bounds=(1.4, 3.4), method=\'bounded\').x\\n    d = cosine_design(u, frequency) @ _fit(cosine_design(ref, frequency), panel[\'v2\'])\\n    return c, d\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train, test = study[\'train\'], study[\'test\']\\n    u = _panel_a_inverse(study[\'panel_a\'], np.concatenate((train[\'x\'], test[\'x\'])))\\n    c, d = _panel_c_features(study[\'panel_c\'], u)\\n    mixed = (c[:, :, None] * d[:, None, :]).reshape(len(u), -1)\\n    prev_u = np.roll(u, 1, axis=1)\\n    next_u = np.roll(u, -1, axis=1)\\n    cyclic = np.column_stack((u**3, u**2 * prev_u, u**2 * next_u, prev_u * u * next_u))\\n    linear = np.column_stack((_intercept(u), c, d, cyclic, mixed))\\n    return dict(n=len(train[\'x\']), linear=linear, mixed=mixed,\\n                eval_times=study[\'eval_times\'], y=train[\'y\'], traj=train[\'traj\'])\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _ridge(x, y, penalty):\\n    gram = x.T @ x\\n    gram[np.diag_indices_from(gram)] += penalty\\n    gram[0, 0] -= penalty\\n    return np.linalg.solve(gram, x.T @ y)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    features = inputs[\'node_00\']\\n    n = features[\'n\']\\n    linear, mixed = features[\'linear\'], features[\'mixed\']\\n    dot = mixed[:, [0, 5, 10, 15]].sum(axis=1)\\n    common = np.column_stack((linear, dot))\\n    slow = np.column_stack((linear, mixed[:, 1]))\\n    times = np.asarray(features[\'eval_times\'])[:, None]\\n\\n    def time_basis(rates):\\n        return -np.expm1(-times * rates)\\n\\n    def residual(rates):\\n        basis = time_basis(rates)\\n        observed = features[\'traj\'].T\\n        return (basis @ np.linalg.lstsq(basis, observed, rcond=None)[0] - observed).ravel()\\n\\n    rates = least_squares(residual, [1.8, 0.4], bounds=([1.0, 0.01], [10.0, 1.0])).x\\n    temporal = time_basis(rates)\\n    amplitudes = np.linalg.lstsq(temporal, features[\'traj\'].T, rcond=None)[0].T\\n    main = common[n:] @ _ridge(common[:n], np.column_stack((features[\'y\'], amplitudes[:, 0])), 0.00015)\\n    tail = slow[n:] @ _ridge(slow[:n], amplitudes[:, 1], 0.0003)\\n    y = main[:, 0]\\n    trajectory = np.column_stack((main[:, 1], tail)) @ temporal.T\\n    end = trajectory[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': trajectory,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), end.mean(), end.std()])}\\n"}'

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
