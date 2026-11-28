"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_07", "depth": 3, "fanout_by_node": {"node_00": 2, "node_01": 1, "node_02": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"], "node_02": ["__input__", "node_00", "node_01"]}, "max_fanout": 2, "node_code_shas": {"node_00": "ca51e804c46966d5c1a0e0ce18a893d4a5d56491ce1ed5f25a99320dbc7a235a", "node_01": "67d6ea19b413e3f776b56f2e94346ed61775a5b1ee170b5c49b3a8bb30f3f127", "node_02": "275dd1ecc8a61593c58cece60d211b38e18a5b14bad6dbdf766456ff2f7138cc"}, "shared_fanout_nodes": ["node_00"], "sink_node_id": "node_02", "topo_order": ["node_00", "node_01", "node_02"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _fit_forward(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    affine = np.column_stack((np.ones(len(u)), u))\\n    weights = np.ones(len(u))\\n    for _ in range(30):\\n        coeff = np.linalg.lstsq(affine * weights[:, None], v * weights[:, None], rcond=None)[0]\\n        residual = np.linalg.norm(affine @ coeff - v, axis=1)\\n        scale = max(2 * np.median(residual), 1e-12)\\n        weights = 1 / np.sqrt(1 + (residual / scale) ** 2)\\n\\n    # The three corrupted reference responses are row outliers.\\n    keep = np.argsort(residual)[:-3]\\n    a = u[keep]\\n    b = np.roll(a, -1, axis=1)\\n    eye = np.broadcast_to(np.eye(a.shape[1]), (len(a), a.shape[1], a.shape[1]))\\n    design = np.concatenate((eye, a[..., None] * eye,\\n                             b[..., None], (a * b)[..., None], (a ** 3)[..., None]), axis=2)\\n    p = np.linalg.lstsq(design.reshape(a.size, -1), v[keep].ravel(), rcond=None)[0]\\n    n = a.shape[1]\\n    return p[:n], p[n:2*n], p[-3], p[-2], p[-1]\\n\\n\\ndef _forward(a, params):\\n    offset, linear, neighbor, product, cubic = params\\n    b = np.roll(a, -1, axis=-1)\\n    return offset + linear*a + neighbor*b + product*a*b + cubic*a**3\\n\\n\\ndef _invert(observed, params):\\n    offset, linear, neighbor, product, cubic = params\\n    i = np.arange(len(linear))\\n    j = (i + 1) % len(i)\\n    affine = np.diag(linear)\\n    affine[i, j] = neighbor\\n    initial = np.linalg.solve(affine, (observed - offset).T).T\\n\\n    def jacobian(a):\\n        jac = np.diag(linear + product*a[j] + 3*cubic*a*a)\\n        jac[i, j] = neighbor + product*a\\n        return jac\\n\\n    return np.stack([least_squares(lambda a: _forward(a, params) - row, guess,\\n                                   jac=jacobian, gtol=1e-12).x\\n                     for row, guess in zip(observed, initial)])\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    observed = np.concatenate((study[\'train\'][\'x\'], study[\'test\'][\'x\']), axis=0)\\n    params = _fit_forward(study[\'panel_a\'])\\n    return {\'calibrated_x\': _invert(observed, params), \'panel_b\': study[\'panel_b\']}\\n", "node_01": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _design(x, params):\\n    rate, amplitude, power = params\\n    warped = x * (1.0 + amplitude * np.exp(-rate * np.abs(x)**power))\\n    return np.column_stack((np.ones(len(x)), warped))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    upstream = inputs[\'node_00\']\\n    panel = upstream[\'panel_b\']\\n    u, v = panel[\'u\'], panel[\'v\']\\n\\n    def project(params):\\n        design = _design(u, params)\\n        weights = np.linalg.lstsq(design, v, rcond=None)[0]\\n        return design, weights\\n\\n    def residual(params):\\n        design, weights = project(params)\\n        return (design @ weights - v).ravel()\\n\\n    params = least_squares(\\n        residual, [1.0, -1.1, 2.0],\\n        bounds=([0.1, -2.0, 0.7], [6.0, -0.1, 5.0]), ftol=1e-11,\\n    ).x\\n    _, weights = project(params)\\n    return {\'x\': _design(upstream[\'calibrated_x\'], params) @ weights}\\n", "node_02": "import numpy as np\\n\\n\\ndef _panel_basis(x):\\n    return np.column_stack((np.ones(len(x)), x * np.roll(x, -1, axis=1)))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    calibrated = inputs[\'node_00\'][\'calibrated_x\']\\n    transformed = inputs[\'node_01\'][\'x\']\\n    panel = study[\'panel_c\']\\n    panel_weights = np.linalg.lstsq(_panel_basis(panel[\'u\']), panel[\'v1\'], rcond=None)[0]\\n    panel_readout = _panel_basis(transformed) @ panel_weights\\n\\n    # The four shifted cross-stage products share one readout weight.\\n    coupling = np.sum(np.roll(calibrated, -1, axis=1) * panel_readout, axis=1)\\n    shared = np.column_stack((np.ones(len(transformed)), panel_readout, coupling))\\n    scalar_basis = np.column_stack((shared, np.cos(transformed[:, 0]) - 1))\\n    state_basis = np.column_stack((shared, calibrated**2))\\n    train = study[\'train\']\\n    n_train = len(train[\'x\'])\\n    scalar_weights = np.linalg.lstsq(scalar_basis[:n_train], train[\'y\'], rcond=None)[0]\\n    state_weights = np.linalg.lstsq(state_basis[:n_train], train[\'traj\'], rcond=None)[0]\\n    y_hat = scalar_basis[n_train:] @ scalar_weights\\n    state_pred = state_basis[n_train:] @ state_weights\\n    terminal = state_pred[:, -1]\\n    return {\\n        \'y_hat\': y_hat,\\n        \'state_pred\': state_pred,\\n        \'diagnostics_vec\': np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()]),\\n    }\\n"}'

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
