"""Recover the calibration, dual operator branches, and two relaxation modes."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-9)[0]
def D(x):return np.c_[np.ones(len(x)),x]
def calibrate(panel,x):
 u,v=panel['u'],panel['v']; rng=np.random.default_rng(45); w=np.roll(u,2,axis=1)
 def gram(a,b,k):return D(np.c_[a,np.tanh(k[0]*a),np.sin(k[1]*b)])
 def coeff(k,g):return np.array([ls(gram(u[g,j],w[g,j],k),v[g,j]) for j in range(4)])
 def fore(z,k,c):return c[:,0]+z*c[:,1]+np.tanh(k[0]*z)*c[:,2]+np.sin(k[1]*np.roll(z,2,axis=1))*c[:,3]
 k=np.array([1.8,2.4]); g=np.arange(len(u)); a=[gram(u[:,j],w[:,j],k) for j in range(4)]
 c=min((coeff(k,i) for i in (rng.choice(len(u),9,False) for _ in range(100))),key=lambda c:np.median(np.max(abs(fore(u,k,c)-v),axis=1)))
 err=np.max(abs(fore(u,k,c)-v),axis=1); g=err<max(.001,6*np.median(err)); k=least_squares(lambda k:(fore(u[g],k,coeff(k,g))-v[g]).ravel(),k,bounds=([1.4,2],[2.2,2.8])).x; c=coeff(k,g)
 z=(x-c[:,0])/c[:,1]; j=np.arange(4)
 for i in range(12):
  J=np.zeros((len(z),4,4)); J[:,j,j]=c[:,1]+k[0]*c[:,2]*(1-np.tanh(k[0]*z)**2);J[:,j,(j+2)%4]=k[1]*c[:,3]*np.cos(k[1]*np.roll(z,2,axis=1));z-=np.linalg.solve(J,(fore(z,k,c)-x)[...,None])[...,0]
 return z


def _features(study, x):
    z = calibrate(study['panel_a'], x)
    panel = study['panel_c']
    u = panel['u']
    def quadratic(x):
        return D(x * np.roll(x, -1, axis=1))
    def cosine(x, freq):
        return D(np.cos(freq*x))
    p = quadratic(z) @ ls(quadratic(u), panel['v1'])
    def loss(freq):
        design = cosine(u, freq)
        return np.mean((design @ ls(design, panel['v2']) - panel['v2'])**2)
    freq = minimize_scalar(loss, bounds=(1.7,3.2), method='bounded', options={'xatol':1e-10}).x
    q = cosine(z, freq) @ ls(cosine(u, freq), panel['v2'])
    return D(np.c_[p, q, np.einsum('ij,ik->ijk', p, q).reshape(len(z),16)])


def _state(times, trajectories, train_features, new_features):
    # The fast mode sees both branches; the slow mode sees the cosine
    # branch and the off-diagonal quadratic/cosine interaction.
    fast_cols = np.arange(11)
    slow_cols = [0, 5, 6, 7, 8, 10]

    def basis(rates):
        return -np.expm1(-times[:, None] * rates)

    def design(features, H):
        return np.column_stack([
            np.einsum('ij,k->ikj', features[:, cols], H[:, j]).reshape(
                len(features) * len(times), len(cols))
            for j, cols in enumerate((fast_cols, slow_cols))
        ])

    observed = trajectories.ravel()

    def residual(rates):
        A = design(train_features, basis(rates))
        return A @ ls(A, observed) - observed

    rates = least_squares(residual, [1.83, .41], bounds=([.85, .03], [5., .84]),
                          ftol=1e-11).x
    H = basis(rates)
    weights = ls(design(train_features, H), observed)
    return (design(new_features, H) @ weights).reshape(len(new_features), len(times))


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    F = _features(study, np.vstack((train['x'], study['test']['x'])))
    diag = F[:, [9,14,19,24]].sum(axis=1)
    design = np.c_[F[:, :9], diag]
    state_design = np.c_[design, F[:, 10]]
    y = design[n:] @ ls(design[:n],train['y'])
    state = _state(np.asarray(study['eval_times']), train['traj'], state_design[:n], state_design[n:])
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': np.array([y.mean(),y.std(),state[:,-1].mean(),state[:,-1].std()])}
