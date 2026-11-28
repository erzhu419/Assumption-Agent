"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_08", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_02": 1, "node_03": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["__input__", "node_00"], "node_02": ["node_00"], "node_03": ["node_01", "node_02"]}, "max_fanout": 2, "node_code_shas": {"node_00": "3667796cc609a1b9aa277d26605a13d297c6ee3bfa02faef1ea63cf995b95e75", "node_01": "db75a2ce02262f764b2ffa7ad56d21a9bd30dddf7148352b0f0c35df2091d4e7", "node_02": "693b0627bf56e790497bcdf3af98080576199214a40db89fa860aefab5524580", "node_03": "0b763cf86b0445cc755d6e6b9ad11e134d442dd54eb5ef5c782c75c657475431"}, "shared_fanout_nodes": ["node_00"], "sink_node_id": "node_03", "topo_order": ["node_00", "node_01", "node_02", "node_03"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _panel(panel, trim=0):\\n    u = _affine(panel[\'u\'])\\n    v = panel[\'v\']\\n    keep = np.arange(len(u))\\n    for _ in range(5 if trim else 0):\\n        c = np.linalg.lstsq(u[keep], v[keep], rcond=None)[0]\\n        keep = np.argsort(np.linalg.norm(u @ c - v, axis=1))[:len(u)-trim]\\n    return np.linalg.lstsq(u[keep], v[keep], rcond=None)[0]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    x, xt = train[\'x\'], study[\'test\'][\'x\']\\n    a, b = _panel(study[\'panel_a\'], 3), _panel(study[\'panel_b\'])\\n\\n    def transform(u):\\n        return _affine(_affine(u) @ a) @ b\\n\\n    w, wt = transform(x), transform(xt)\\n    z, zt = (w-w.mean(0))/w.std(0), (wt-w.mean(0))/w.std(0)\\n    return {\'z\': z, \'zt\': zt, \'traj\': train[\'traj\']}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _affine(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _calibration(panel, trim=0):\\n    design, values = _affine(panel[\'u\']), panel[\'v\']\\n    keep = np.arange(len(design))\\n    for _ in range(5 if trim else 0):\\n        coef = np.linalg.lstsq(design[keep], values[keep], rcond=None)[0]\\n        keep = np.argsort(np.linalg.norm(design @ coef - values, axis=1))[:-trim]\\n    return np.linalg.lstsq(design[keep], values[keep], rcond=None)[0]\\n\\n\\ndef _harmonics(x, frequency):\\n    phase = x * frequency\\n    return np.column_stack((_affine(x), np.sin(phase), np.cos(phase)))\\n\\n\\ndef _panel_frequencies(panel):\\n    u, values = panel[\'u\'], panel[\'v1\']\\n\\n    def residual(frequency):\\n        design = _harmonics(u, frequency)\\n        coef = np.linalg.lstsq(design, values, rcond=None)[0]\\n        return (design @ coef - values).ravel()\\n\\n    fits = (least_squares(residual, np.full(4, start), bounds=(0.15, 8),\\n                          max_nfev=150) for start in (1.5, 2.5, 3.5, 4.5, 5.5))\\n    return min(fits, key=lambda fit: fit.cost).x\\n\\n\\ndef _kernel(x, u, w, v, harmonics, reference):\\n    dot = x @ u.T\\n    quadratic = (dot**2 + (x*x) @ (u*u).T) / 2\\n    distance = np.sum((w[:, None] - v[None, :])**2, axis=-1)\\n    return 1 + dot + quadratic + np.exp(-distance) + 0.1 * (harmonics @ reference.T)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    shared = inputs[\'node_00\']\\n    x, xt = study[\'train\'][\'x\'], study[\'test\'][\'x\']\\n    # Restore panel coordinates: the harmonic frequencies are in panel units.\\n    a, b = _calibration(study[\'panel_a\'], 3), _calibration(study[\'panel_b\'])\\n    calibrated = _affine(_affine(x) @ a) @ b\\n    scale, center = calibrated.std(0), calibrated.mean(0)\\n    w = shared[\'z\'] * scale + center\\n    wt = shared[\'zt\'] * scale + center\\n    frequency = _panel_frequencies(study[\'panel_c\'])\\n    h, ht = _harmonics(w, frequency), _harmonics(wt, frequency)\\n    gram = _kernel(x, x, w, w, h, h)\\n    gram.flat[::len(x)+1] += 0.002\\n    weights = np.linalg.solve(gram, study[\'train\'][\'y\'])\\n    return {\'y_hat\': _kernel(xt, x, wt, w, ht, h) @ weights}\\n", "node_02": "import numpy as np\\n\\n\\ndef _kernel(x, u):\\n    dot = x @ u.T\\n    distance = np.maximum((x*x).sum(1)[:, None] + (u*u).sum(1) - 2*dot, 0)\\n    quad = (dot*dot + (x*x) @ (u*u).T) / 2\\n    return 1 + dot + quad + np.exp(-0.1*distance)\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    state = inputs[\'node_00\']\\n    z, zt, traj = state[\'z\'], state[\'zt\'], state[\'traj\']\\n    gram = _kernel(z, z)\\n    gram.flat[::len(z)+1] += 0.003\\n    weights = np.linalg.solve(gram, traj)\\n    return {\'trajectories\': (_kernel(zt, z) @ weights)[None, :, :]}\\n", "node_03": "import numpy as np\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    y_hat = inputs[\'node_01\'][\'y_hat\']\\n    trajectories = inputs[\'node_02\'][\'trajectories\']\\n    state_pred = trajectories.mean(axis=0)\\n    terminal = state_pred[:, -1]\\n    return {\'y_hat\': y_hat, \'state_pred\': state_pred,\\n            \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])}\\n"}'

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
