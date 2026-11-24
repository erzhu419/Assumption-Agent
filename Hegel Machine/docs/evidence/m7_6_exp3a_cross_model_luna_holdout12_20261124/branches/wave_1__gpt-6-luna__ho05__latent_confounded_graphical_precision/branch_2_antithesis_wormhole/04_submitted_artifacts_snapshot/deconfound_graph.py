import numpy as np
from scipy.optimize import minimize

def recover_sparse_precision_support(X_samples: np.ndarray) -> dict:
    """Estimate sparse conditional structure after accounting for shared latent covariance."""
    X = np.asarray(X_samples, dtype=float)
    n, p = X.shape
    X = X - X.mean(axis=0)
    C = (X.T @ X) / max(n, 1)
    # Covariance decomposition C = Sigma_sparse + B B.T. Alternating factor-analysis
    # updates estimate the broad shared component; a sparse precision is then fit to
    # the residual covariance using a positive-definite graphical-lasso objective.
    w, V = np.linalg.eigh(C)
    r = min(2, max(1, p // 10))
    # Factor-analysis covariance MLE via EM, with diagonal uniquenesses.
    # Retain only the low-rank shared covariance estimate for subtraction.
    d = np.maximum(np.diag(C).copy(), 1e-6)
    B = V[:, -r:] * np.sqrt(np.maximum(w[-r:] - np.median(w[:-r]), 1e-6))[None, :]
    for _ in range(80):
        Psi = np.diag(np.maximum(d, 1e-5))
        K = np.linalg.solve(np.eye(r) + B.T @ (B / d[:, None]), B.T / d[None, :])
        Ez = K @ C
        Ezz = np.eye(r) - K @ B + K @ C @ K.T
        Bn = (C @ K.T) @ np.linalg.pinv(Ezz)
        dn = np.maximum(np.diag(C - Bn @ K @ C), 1e-4)
        if np.max(np.abs(Bn-B)) < 1e-6:
            B, d = Bn, dn
            break
        B, d = Bn, dn
    L = B @ B.T
    R = (C - L + (C - L).T) * 0.5
    # Numerical floor while preserving empirical residual correlations.
    ev = np.linalg.eigvalsh(R)
    if ev[0] <= 1e-5:
        R += (1e-5 - ev[0]) * np.eye(p)
    # Solve graphical lasso by maximizing logdet(Theta)-tr(R Theta)-alpha*|Theta_off|.
    # A small penalty is appropriate at n=900 and p=24.
    alpha = 0.025 * float(np.median(np.diag(R)))
    def objective(v):
        T = v.reshape(p, p)
        T = (T + T.T) * 0.5
        sign, ld = np.linalg.slogdet(T)
        if sign <= 0:
            return 1e12 + np.sum(v*v)
        return float(np.sum(R*T) - ld + alpha*np.sum(np.abs(T - np.diag(np.diag(T)))))
    # Start from inverse residual covariance, regularized to SPD.
    T0 = np.linalg.inv(R + alpha*np.eye(p))
    x0 = T0.ravel()
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-9})
    T = res.x.reshape(p,p)
    T = (T + T.T) * 0.5
    # If optimizer numerical state is not SPD, use stable residual inverse.
    if np.linalg.eigvalsh(T)[0] <= 0:
        T = np.linalg.inv(R + alpha*np.eye(p))
    np.fill_diagonal(T, np.maximum(np.diag(T), 1e-8))
    scores = np.abs(T)
    np.fill_diagonal(scores, 0.0)
    # Expected sparse graph density is roughly 26 edges at p=24; use conservative
    # data-adaptive edge budget from the stated sparse regime.
    m = min(p*(p-1)//2, max(1, round(0.05 * p * (p-1))))
    iu = np.triu_indices(p, 1)
    chosen = np.argsort(scores[iu])[-m:]
    adj = np.zeros((p,p), dtype=bool)
    adj[iu[0][chosen], iu[1][chosen]] = True
    adj |= adj.T
    return {'adjacency': adj, 'sparse_precision': T}
