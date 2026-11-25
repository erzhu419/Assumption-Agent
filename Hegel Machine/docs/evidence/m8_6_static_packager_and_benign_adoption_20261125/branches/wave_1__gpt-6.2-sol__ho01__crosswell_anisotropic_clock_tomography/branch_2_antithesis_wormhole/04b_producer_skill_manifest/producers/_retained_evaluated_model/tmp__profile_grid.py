import numpy as np
from scipy.special import logsumexp

def location(rays, a, y, keep, initial, width, height, prior_width, marginal):
    # Condition on screened picks; project out the background and clock gauge.
    keep = keep.ravel()
    design = a[keep]
    q = np.linalg.qr(design, mode='reduced')[0]
    yy = y[keep]
    res = yy - q @ (q.T @ yy)
    sigma = max(.015, np.median(abs(res - initial[2] * (rays.plume(*initial[:2])[keep] - q @ (q.T @ rays.plume(*initial[:2])[keep])))) / .67449)
    xs = np.linspace(0, width, int(np.ceil(width / .65)) + 1)
    zs = np.linspace(0, height, int(np.ceil(height / .65)) + 1)
    x, z = (v.ravel() for v in np.meshgrid(xs,zs))
    # Use weak fractional-contrast regularization, not a spatial cutoff.
    logamp = np.linspace(np.log(.008),np.log(.65),100)
    amp = np.exp(logamp)
    ref = .17 * initial[3]
    prior = -.5 * ((logamp - np.log(ref)) / prior_width)**2
    score = np.empty(len(x))
    for i in range(0,len(x),128):
        j = slice(i,i+128)
        k = rays.plume(x[j,None], z[j,None])[:, keep]
        f = k - (k @ q) @ q.T
        ss = np.sum(f*f,axis=1)
        ty = f @ res
        ls = (ty[:,None]*amp-.5*ss[:,None]*amp**2)/sigma**2 + prior
        score[j] = logsumexp(ls, axis=1) if marginal else np.max(ls,axis=1)
    if marginal:
        mass = np.exp(score-logsumexp(score)).reshape(len(zs),len(xs))
        mass[[0,-1],:] *= .5
        mass[:,[0,-1]] *= .5
        mass = mass.ravel()/mass.sum()
        return np.array([mass @ x, mass @ z])
    j=np.argmax(score)
    return np.array([x[j],z[j]])
