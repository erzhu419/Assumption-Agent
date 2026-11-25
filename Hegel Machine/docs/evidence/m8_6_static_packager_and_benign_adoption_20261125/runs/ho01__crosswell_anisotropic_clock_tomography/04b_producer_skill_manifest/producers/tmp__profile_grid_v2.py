import numpy as np
from scipy.special import logsumexp, roots_legendre

_NODES, _WEIGHTS = roots_legendre(24)

def location(rays, a, y, keep, initial, width, height, prior_width):
    keep = keep.ravel()
    q = np.linalg.qr(a[keep], mode='reduced')[0]
    def project(v):
        return v - q @ (q.T @ v)
    res = project(y[keep])
    k, kx, kz = rays.plume(*initial[:2], derivatives=True)
    e = res - initial[2] * project(k[keep])
    sigma = max(.012, np.median(abs(e - np.median(e))) / .67449)
    jac = np.column_stack((initial[2]*kx[keep], initial[2]*kz[keep], k[keep]))
    svd_u, singular, axes = np.linalg.svd(project(jac), full_matrices=False)
    uncertainty = np.linalg.norm(axes.T[:2] * (sigma / np.maximum(singular, 1e-12)), ord=2)
    # Avoid quantizing well-resolved likelihood peaks on the edge-search grid.
    if uncertainty < .40 and min(initial[0], width-initial[0], initial[1], height-initial[1]) > 2:
        return initial[:2]
    xs = np.linspace(0, width, int(np.ceil(width / .55)) + 1)
    zs = np.linspace(0, height, int(np.ceil(height / .55)) + 1)
    x, z = (v.ravel() for v in np.meshgrid(xs, zs))
    score = np.empty(len(x))
    logref = np.log(max(.008, .17 * initial[3]))
    for i in range(0, len(x), 128):
        j = slice(i, i+128)
        k = rays.plume(x[j,None], z[j,None])[:, keep]
        f = k - (k @ q) @ q.T
        ss = np.maximum(np.sum(f*f, axis=1), 1e-14)
        ty = f @ res
        mean, sd = ty / ss, sigma / np.sqrt(ss)
        lo = np.clip(mean-8*sd, .003, .79)
        hi = np.clip(mean+8*sd, lo+1e-7, .8)
        half = .5 * (hi - lo)
        amp = (lo+half)[:,None] + half[:,None]*_NODES
        ls = ((ty[:,None]*amp - .5*ss[:,None]*amp**2)/sigma**2
              - .5*((np.log(amp)-logref)/prior_width)**2 - np.log(amp))
        score[j] = logsumexp(ls, b=_WEIGHTS, axis=1) + np.log(half)
    mass = np.exp(score - np.max(score)).reshape(len(zs),len(xs))
    mass[[0,-1],:] *= .5
    mass[:,[0,-1]] *= .5
    mass = mass.ravel() / mass.sum()
    return np.array([mass @ x, mass @ z])
