"""Fit separate scalar and trajectory models from the frozen DAG payloads."""
import numpy as np


def _design(train, test, nterms, target):
    center = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), .1)
    a, b = (train-center)/scale, (test-center)/scale
    left, right = np.triu_indices(a.shape[1])
    p, v = a[:, left]*a[:, right], b[:, left]*b[:, right]
    offset = p.mean(axis=0)
    spread = np.maximum(p.std(axis=0), .1)
    p, v = (p-offset)/spread, (v-offset)/spread
    if nterms == -1:
        return np.c_[a, p], np.c_[b, v]
    if nterms == 0:
        return a, b
    q = np.linalg.qr(np.c_[np.ones(len(a)), a], mode='reduced')[0]
    residual = target - q@(q.T@target)
    orth = p - q@(q.T@p)
    selected = []
    for _ in range(nterms):
        norms = np.linalg.norm(orth, axis=0)
        scores = np.linalg.norm(orth.T@residual, axis=1)/np.maximum(norms, 1e-10)
        scores[norms < 1e-10] = -np.inf
        j = int(np.argmax(scores))
        if not np.isfinite(scores[j]):
            break
        selected.append(j)
        unit = orth[:, j]/norms[j]
        residual = residual - np.outer(unit, unit@residual)
        orth = orth - np.outer(unit, unit@orth)
        orth[:, j] = 0
    return np.c_[a, p[:, selected]], np.c_[b, v[:, selected]]


def _predict(train, test, target, nterms, penalty):
    a, b = _design(train, test, nterms, target)
    center = target.mean(axis=0)
    gram = a.T@a + penalty*np.eye(a.shape[1])
    coef = np.linalg.solve(gram, a.T@(target-center))
    return center + b@coef


def _choose(train, test, target, reconstruct, observed):
    n = len(train)
    if n < 12 or np.max(np.std(train, axis=0)) < 1e-10:
        return None
    menu = [(k, p) for k in (0, 1, 2, 4, -1) for p in (1., 4., 16., 48.)]
    rng = np.random.default_rng(187)
    losses = [[] for _ in menu]
    rows = np.arange(n)
    for _ in range(3):
        for va in np.array_split(rng.permutation(n), 4):
            tr = np.setdiff1d(rows, va)
            for j, (k, penalty) in enumerate(menu):
                pred = _predict(train[tr], train[va], target[tr], k, penalty)@reconstruct
                losses[j].append(np.mean(np.abs(pred - observed[va])))
    losses = np.asarray(losses)
    score = .8*losses.mean(axis=1) + .2*np.quantile(losses, .85, axis=1)
    for j in np.argsort(score, kind='stable'):
        k, penalty = menu[j]
        pred = _predict(train, test, target, k, penalty)@reconstruct
        if np.all(np.isfinite(pred)):
            return pred
    return None


def run_node(inputs: dict) -> dict:
    w = np.asarray(inputs['node_00']['calibrated'], dtype=float)
    data = inputs['node_01']
    target = np.asarray(data['targets'], dtype=float)
    basis = np.asarray(data['basis'], dtype=float)
    fallback = np.asarray(inputs['node_02']['fit'], dtype=float)
    n = len(target)
    q = np.linalg.qr(basis, mode='reduced')[0]
    y = _choose(w[:n], w[n:], target[:, :1], np.ones((1, 1)), target[:, :1])
    state = _choose(w[:n], w[n:], target[:, 1:]@q, q.T, target[:, 1:])
    y = fallback[:, 0] if y is None else y[:, 0]
    state = (fallback[:, 1:]@q)@q.T if state is None else state
    tail = state[:, -1]
    summary = np.array([y.mean(), y.std(), tail.mean(), tail.std()]) if len(y) else np.zeros(4)
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': summary}
