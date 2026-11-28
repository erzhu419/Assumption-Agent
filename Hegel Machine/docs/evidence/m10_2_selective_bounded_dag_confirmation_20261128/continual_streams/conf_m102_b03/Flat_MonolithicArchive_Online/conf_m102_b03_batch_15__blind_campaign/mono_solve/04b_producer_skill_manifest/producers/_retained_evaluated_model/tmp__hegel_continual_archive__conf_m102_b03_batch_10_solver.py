import numpy as np
from scipy.optimize import least_squares

def D(x): return np.c_[np.ones(len(x)),x]
def ls(x,y): return np.linalg.lstsq(x,y,rcond=1e-10)[0]

def calibrate(panel,x):
 u,v=panel['u'],panel['v']; l=D(u); rng=np.random.default_rng(45)
 a=min((ls(l[i],v[i]) for i in (rng.choice(len(u),8,False) for _ in range(240))),key=lambda a:np.median(np.linalg.norm(l@a-v,axis=1)))
 r=np.linalg.norm(l@a-v,axis=1); good=r<5*np.median(r)
 a=ls(l[good],v[good]); w=np.roll(u,-1,axis=1)
 b,d,e,f,g=np.stack([ls(D(np.c_[u[:,j],w[:,j],u[:,j]*w[:,j],u[:,j]**3])[good],v[good,j]) for j in range(4)],axis=1)
 z=np.linalg.solve(a[1:].T,(x-a[0]).T).T; j=np.arange(4)
 for _ in range(20):
  w=np.roll(z,-1,axis=1); r=b+d*z+e*w+f*z*w+g*z**3-x
  if not r.size or abs(r).max()<1e-12: break
  J=np.zeros((len(z),4,4)); J[:,j,j]=d+f*w+3*g*z*z; J[:,j,(j+1)%4]=e+f*z
  z-=np.linalg.solve(J,r[...,None])[...,0]
 return z


from scipy.optimize import minimize_scalar


def _panel_features(panel, z):
    """Fit the two auxiliary maps from panel observations only."""
    u = panel['u']
    def left(a, k):
        w = k * a
        return D(np.c_[a, w / (1 + w*w), a**3])
    def right(a, k):
        return D(np.c_[a, np.cos(k*a)])
    def predict(design, v, bounds):
        def loss(k):
            m = design(u, k)
            return np.mean((m @ ls(m, v) - v)**2)
        k = minimize_scalar(loss, bounds=bounds, method='bounded').x
        return design(z, k) @ ls(design(u, k), v)
    a = predict(left, panel['v1'], (.5, 2.0))
    c = predict(right, panel['v2'], (1.6, 3.0))
    dot = np.sum(a*c, axis=1)
    cross = a[:, 0] * c[:, 1]
    return D(np.c_[a, c, dot]), D(np.c_[a, c, dot, cross]), D(np.c_[c, cross])


def _trajectory(t, observed, fast, slow):
    """Profile out spatial amplitudes while fitting two shared rates."""
    n, nt = observed.shape
    def design(p, f, s):
        h = -np.expm1(-t[:, None] * p)
        return np.concatenate((f[:, None, :] * h[None, :, :1],
                               s[:, None, :] * h[None, :, 1:]), axis=2).reshape(len(f)*nt, f.shape[1]+s.shape[1])
    target = observed.ravel()
    def residual(p):
        m = design(p, fast[:n], slow[:n])
        return m @ ls(m, target) - target
    p = least_squares(residual, [1.8, .4], bounds=([.8, .025], [6., .79]), ftol=1e-10).x
    m = design(p, fast[:n], slow[:n])
    return (design(p, fast[n:], slow[n:]) @ ls(m, target)).reshape(len(fast)-n, nt)


def solve_batch(study: dict) -> dict:
    tr = study['train']
    n = len(tr['x'])
    x = np.vstack((tr['x'], study['test']['x']))
    z = calibrate(study['panel_a'], x)
    scalar, fast, slow = _panel_features(study['panel_c'], z)
    y = scalar[n:] @ ls(scalar[:n], tr['y'])
    state = _trajectory(study['eval_times'], tr['traj'], fast, slow)
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
