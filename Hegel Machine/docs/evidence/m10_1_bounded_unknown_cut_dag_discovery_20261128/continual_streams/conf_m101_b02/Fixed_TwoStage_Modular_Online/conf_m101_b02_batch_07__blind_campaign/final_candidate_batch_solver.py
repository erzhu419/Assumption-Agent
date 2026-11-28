"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_10", "depth": 1, "fanout_by_node": {"node_00": 0}, "incoming": {"node_00": ["__input__"]}, "max_fanout": 0, "node_code_shas": {"node_00": "444d3bbf338abf7adb6a7bb32a98d74829cf9394bf23cb1121fe053ac72d6b3b"}, "shared_fanout_nodes": [], "sink_node_id": "node_00", "topo_order": ["node_00"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\nfrom scipy.optimize import least_squares\\n\\n\\ndef _ls(x, y):\\n    return np.linalg.lstsq(x, y, rcond=1e-9)[0]\\n\\n\\ndef _aff(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _cross(x):\\n    return (x[:, [0, 0, 1, 2]] * x[:, [1, 3, 2, 3]])\\n\\n\\ndef _profile(fun, x, y, start, bounds):\\n    def residual(p):\\n        d = fun(x, p)\\n        return (d @ _ls(d, y) - y).ravel()\\n    p = least_squares(residual, start, bounds=bounds, max_nfev=100).x\\n    return p, _ls(fun(x, p), y)\\n\\n\\ndef _calibrate(study):\\n    a, b, c = (study[\'panel_\' + k] for k in \'abc\')\\n    ad = _aff(a[\'u\'])\\n    aw = np.vstack((np.zeros(4), np.eye(4)))\\n    for _ in range(10):\\n        r = ad @ aw - a[\'v\']\\n        w = 1 / np.sqrt(1 + np.sum(r*r, axis=1) / .02**2)\\n        aw = _ls(ad*w[:, None], a[\'v\']*w[:, None])\\n    def odd(z, p):\\n        return z*(1-p[1]*np.exp(-p[0]*z*z))\\n    def bd(z, p):\\n        return _aff(np.c_[odd(z, p), z*z])\\n    def cd(z, p):\\n        return _aff(np.c_[z, np.sqrt(1+p[0]*z*z)])\\n    bp, bw = _profile(bd, b[\'u\'], b[\'v\'], [1., 1.1], ([.1, .2], [5., 2.]))\\n    cp, cw = _profile(cd, c[\'u\'], c[\'v2\'], [.9], ([.01], [4.]))\\n    pw = _ls(_aff(_cross(c[\'u\'])), c[\'v1\'])\\n    return aw, lambda z: bd(z, bp) @ bw, lambda z: _aff(_cross(z)) @ pw, lambda z: cd(z, cp) @ cw, lambda z: odd(z, bp)\\n\\n\\ndef _readout_gram(ref, query, config):\\n    mode, length, interaction, fitted = config\\n    z, g = ref\\n    v, h = query\\n    nz = len(z)\\n    a = np.r_[z, v]\\n    b = np.r_[g, h]\\n    # Fit normalization on the training features unless using the batch-design variant.\\n    za = a[:nz] if fitted else a\\n    zb = b[:nz] if fitted else b\\n    a = a / np.maximum(za.std(0), 1e-8)\\n    b = (b-zb.mean(0)) / np.maximum(zb.std(0), 1e-8)\\n    lin = a @ a[:nz].T / 4\\n    joint = b @ b[:nz].T / b.shape[1]\\n    parts = []\\n    for j in range(4):\\n        w = b[:, j::4]\\n        dot = w @ w[:nz].T / w.shape[1]\\n        if mode == \'poly\':\\n            parts.append((1+dot/length)**3-1)\\n        else:\\n            norm = np.mean(w*w, axis=1)\\n            r2 = np.maximum(norm[:, None]+norm[None, :nz]-2*dot, 0) / length**2\\n            if mode == \'rbf\':\\n                parts.append(np.exp(-r2/2))\\n            else:\\n                r = np.sqrt(r2)\\n                parts.append((1+r)*np.exp(-r))\\n    kernel = 1 + lin + sum(parts)/4\\n    if interaction == \'mixed\':\\n        kernel += (.1*lin + .02*lin**2)*joint\\n    else:\\n        from itertools import combinations\\n        kernel += sum(parts[i]*parts[j] for i,j in combinations(range(4),2)) * (.05/6)\\n        kernel += sum(parts[i]*parts[j]*parts[k] for i,j,k in combinations(range(4),3)) * (.02/4)\\n    return kernel[:nz], kernel[nz:]\\n\\n\\ndef _readout_features(x, group, aw, odd):\\n    coord, lift = group\\n    if coord == 1:\\n        x = np.linalg.solve(aw[1:].T, (x-aw[0]).T).T\\n    elif coord == 2:\\n        x = x @ aw[1:] + aw[0]\\n    f = x**3 if lift == \'cube\' else odd(x)\\n    return x, np.c_[x, f]\\n\\n\\ndef _predict_readout(ref, query, y, cfg):\\n    _, mode, length, interaction, fitted, reg = cfg\\n    k, q = _readout_gram(ref, query, (mode, length, interaction, fitted))\\n    return q @ np.linalg.solve(k + reg*np.eye(len(k)), y)\\n\\n\\ndef _predict(study, specs):\\n    aw, _, _, _, odd = _calibrate(study)\\n    tr = study[\'train\']\\n    x, q = tr[\'x\'], study[\'test\'][\'x\']\\n    results = []\\n    for choices, target in zip(specs, (tr[\'y\'], tr[\'traj\'])):\\n        pred = []\\n        for cfg in choices:\\n            ref = _readout_features(x, cfg[0], aw, odd)\\n            query = _readout_features(q, cfg[0], aw, odd)\\n            pred.append(_predict_readout(ref, query, target, cfg))\\n        results.append(np.mean(pred, axis=0))\\n    return results\\n\\n\\n_READOUTS = [[((1, \'cube\'), \'matern\', 8.0, \'mixed\', False, 1e-05), ((1, \'cube\'), \'matern\', 12.0, \'mixed\', False, 0.0003)], [((0, \'odd\'), \'matern\', 6.0, \'mixed\', False, 0.0001), ((1, \'odd\'), \'poly\', 12.0, \'limited\', True, 0.0001)]]\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs.get(\'__input__\', inputs)\\n    y, state = _predict(study, _READOUTS)\\n    terminal = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])}\\n"}'

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
