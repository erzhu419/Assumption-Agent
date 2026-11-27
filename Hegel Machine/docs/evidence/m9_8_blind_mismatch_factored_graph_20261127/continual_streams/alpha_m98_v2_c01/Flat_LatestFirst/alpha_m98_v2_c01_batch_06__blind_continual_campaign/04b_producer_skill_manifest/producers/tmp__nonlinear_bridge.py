import numpy as np


def fit_bridge(cal):
    a = np.asarray(cal['true_channel_loads'], float)
    b = np.asarray(cal['bridge_raw'], float)
    nxt = [1, 2, 3, 0]
    prv = [3, 0, 1, 2]
    # Gross bridge errors are large even when their sum cancels.
    center = np.median(b, axis=0)
    spread = np.median(abs(b-center), axis=0)
    keep = np.max(abs(b-center)/np.maximum(spread, .1), axis=1) < 12
    co = np.array([np.linalg.lstsq(np.column_stack((np.ones(len(a)), a[:,j], a[:,nxt[j]], a[:,prv[j]], a[:,j]*a[:,nxt[j]], a[:,j]**3))[keep], b[keep,j], rcond=None)[0] for j in range(4)])
    off, lin, cross_next, cross_prev, pair, cubic = co.T
    linear = np.diag(lin)
    linear[np.arange(4),nxt] = cross_next
    linear[np.arange(4),prv] = cross_prev

    def invert(raw):
        y = np.asarray(raw, float)-off
        q = np.linalg.solve(linear, y.reshape(-1,4).T).T
        y = y.reshape(-1,4)
        for _ in range(24):
            f = q@linear.T + pair*q*q[:,nxt] + cubic*q**3-y
            if np.max(abs(f)) < 1e-11:
                break
            jac = np.broadcast_to(linear,(len(q),4,4)).copy()
            j = np.arange(4)
            jac[:,j,j] += pair*q[:,nxt]+3*cubic*q*q
            jac[:,j,nxt] += pair*q
            step = np.linalg.solve(jac, f[...,None])[...,0]
            q -= step/np.maximum(1.,np.max(abs(step),axis=1,keepdims=True))
        return q.reshape(raw.shape)
    return invert, keep
