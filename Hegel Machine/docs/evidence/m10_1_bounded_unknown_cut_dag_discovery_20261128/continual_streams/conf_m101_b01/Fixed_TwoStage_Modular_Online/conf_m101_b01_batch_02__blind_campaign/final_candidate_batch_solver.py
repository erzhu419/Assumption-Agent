"""Compiled ExecutableComputationDAG runtime (`M10.1` instrumented anti-aliasing & `exec_count == 1`)."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
import numpy as np

_DAG_SPEC = '{"dag_id": "dag_03", "depth": 2, "fanout_by_node": {"node_00": 1, "node_01": 0}, "incoming": {"node_00": ["__input__"], "node_01": ["node_00"]}, "max_fanout": 1, "node_code_shas": {"node_00": "aaf75fc06dc271b90af448d3ac4023d892ad428f7f8d658060f7283f3ff1a604", "node_01": "a317fa5d14c65e5e03681db5cab3623de5caa73bf3f4fbda4cd6ce03c552680f"}, "shared_fanout_nodes": [], "sink_node_id": "node_01", "topo_order": ["node_00", "node_01"]}'
_NODE_SOURCES = '{"node_00": "import numpy as np\\n\\n\\ndef _calibration(panel):\\n    u, v = panel[\'u\'], panel[\'v\']\\n    rng = np.random.default_rng(452)\\n    best = (np.inf, None, None)\\n    for _ in range(100):\\n        i, j = rng.choice(len(u), 2, replace=False)\\n        d = u[j] - u[i]\\n        if np.min(np.abs(d)) < 0.06:\\n            continue\\n        gain = (v[j] - v[i]) / d\\n        offset = v[i] - gain * u[i]\\n        err = np.linalg.norm(u * gain + offset - v, axis=1)\\n        score = np.median(err)\\n        if score < best[0]:\\n            best = (score, gain, offset)\\n    _, gain, offset = best\\n    err = np.linalg.norm(u * gain + offset - v, axis=1)\\n    keep = err <= max(0.001, 5 * np.median(err))\\n    uc, vc = u[keep], v[keep]\\n    a = uc - uc.mean(axis=0)\\n    gain = np.sum(a * (vc - vc.mean(axis=0)), axis=0) / np.sum(a * a, axis=0)\\n    return gain, vc.mean(axis=0) - uc.mean(axis=0) * gain\\n\\n\\ndef _affine_features(x):\\n    return np.column_stack((np.ones(len(x)), x))\\n\\n\\ndef _coordinates(study, x):\\n    gain, offset = _calibration(study[\'panel_a\'])\\n    panel = study[\'panel_b\']\\n    operator = np.linalg.lstsq(_affine_features(panel[\'u\']), panel[\'v\'], rcond=None)[0]\\n    return _affine_features((x - offset) / gain) @ operator\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    study = inputs[\'__input__\']\\n    train = study[\'train\']\\n    z = _coordinates(study, np.concatenate((train[\'x\'], study[\'test\'][\'x\'])))\\n    n = len(train[\'x\'])\\n    return {\'z_tr\': z[:n], \'z_te\': z[n:], \'train_y\': train[\'y\'],\\n            \'train_traj\': train[\'traj\'], \'eval_times\': study[\'eval_times\']}\\n", "node_01": "import numpy as np\\n\\n\\ndef _features(z):\\n    a, b, c, d = z.T\\n    return np.column_stack((np.ones(len(z)), a*b, a*d, b*c, c*d))\\n\\n\\ndef _readout_features(base):\\n    q = base[:, 1:]\\n    # q0*q3 and q1*q2 are the same monomial; include it only once.\\n    quadratic = [q[:, i]*q[:, j] for i in range(4) for j in range(i, 4)\\n                 if (i, j) != (1, 2)]\\n    return np.column_stack((base, *quadratic))\\n\\n\\ndef run_node(inputs: dict) -> dict:\\n    data = inputs[\'node_00\']\\n    train = _features(data[\'z_tr\'])\\n    test = _features(data[\'z_te\'])\\n    y_coef = np.linalg.lstsq(_readout_features(train), data[\'train_y\'], rcond=None)[0]\\n    state_coef = np.linalg.lstsq(train, data[\'train_traj\'], rcond=None)[0]\\n    y = _readout_features(test) @ y_coef\\n    state = test @ state_coef\\n    end = state[:, -1]\\n    return {\'y_hat\': y, \'state_pred\': state,\\n            \'diagnostics_vec\': np.array([y.mean(), y.std(), end.mean(), end.std()])}\\n"}'

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
