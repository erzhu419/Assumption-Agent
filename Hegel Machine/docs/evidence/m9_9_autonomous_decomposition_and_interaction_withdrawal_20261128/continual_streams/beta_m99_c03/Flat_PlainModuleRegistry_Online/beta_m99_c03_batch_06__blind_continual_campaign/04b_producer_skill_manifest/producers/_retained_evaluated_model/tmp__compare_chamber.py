import sys, json, pickle
import numpy as np
from itertools import product
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import splu
sys.path.insert(0, '/holdout_task')
from _verified_upstream import calibrate_upstream


def axis(kappa, velocity, length, robin, wall):
    n = len(kappa)
    h = length / (n - 1)
    i = np.arange(n)
    prev, nxt = (i - 1) % n, (i + 1) % n
    left = (kappa + kappa[prev]) / (2*h*h) + np.maximum(velocity, 0) / h
    right = (kappa + kappa[nxt]) / (2*h*h) + np.maximum(-velocity, 0) / h
    leak = np.zeros(n)
    if wall != 'P':
        left[0] = right[-1] = 0
        right[0], left[-1] = kappa[[0, -1]] / (h*h)
        if wall == 'R':
            leak[[0, -1]] = robin / h
    return sparse.csc_matrix((np.r_[left + right + leak, -left, -right], (np.tile(i, 3), np.r_[i, prev, nxt])), shape=(n,n))


def forward(s, walls):
    nx, ny = s['nx'], s['ny']
    ax, ay = [axis(s['kappa_'+d], s['vel_'+d], s['L'+d], s['robin_'+d], w) for d,w in zip('xy',walls)]
    op = sparse.kron(sparse.eye(ny),ax,format='csc') + sparse.kron(ay,sparse.eye(nx),format='csc') + sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
    ident = sparse.eye(nx*ny,format='csc')
    rhs = np.column_stack([u.ravel(order='F') for u in s['source_modes']])
    i,j = s['sensor_ij'].T
    design = []
    for mu in s['mu_shifts']:
        fields = splu(op+mu*ident).solve(rhs)
        design.append(fields[i+nx*j])
    return np.vstack(design), fields.T.reshape(-1,ny,nx).transpose(0,2,1)


def fit(S,y,Z,s):
    cols = np.column_stack((S,y))
    coef = np.linalg.solve(Z.T@Z + s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T@cols)
    res = cols-Z@coef
    A,b = res[:,:-1],res[:,-1]
    n = S.shape[1]; r = s['source_ridge']; B = s['total_budget']; C = s['u_box']
    f = lambda w: 0.5*(np.linalg.norm(A@w-b)**2 + r*(w@w))
    g = lambda w: A.T@(A@w-b) + r*w
    ans = minimize(f, np.full(n,B/n), jac=g, bounds=[(0,C)]*n, constraints=[{'type':'eq','fun':lambda w:w.sum()-B,'jac':lambda w:np.ones(n)}], method='SLSQP',options={'ftol':1e-13})
    assert ans.success, ans.message
    return ans.x, coef[:,-1]-coef[:,:-1]@ans.x, f(ans.x)

if __name__ == '__main__':
    item = pickle.load(open('public_batch.pkl','rb'))[0]
    raw = item['study']; payload = json.load(open('_upstream_param_payload.json'))
    sweep = raw['sensor_calibration_sweeps']; x = sweep['ref_rosette']; v = sweep['raw_rosette']; good = (np.abs(v)<=8.5).all(axis=1)
    X = np.column_stack((np.ones(sum(good)), x[good])); affine = np.linalg.lstsq(X,v[good],rcond=None)[0]
    print('full rosette affine rows (offset, reference channels):',np.round(affine,6)); print('rosette fit RMS per channel',np.sqrt(np.mean((X@affine-v[good])**2,axis=0)))
    s = calibrate_upstream(raw,payload); y = s['y_obs_raw']; Z = s['Z_nuisance']; groups=np.arange(len(y))//4
    for p in product('RNP',repeat=2):
        S,_=forward(s,p); w,gamma,obj=fit(S,y,Z,s); loss=[]
        for group in np.unique(groups):
            valid=groups==group; train=~valid
            wc,gc,_=fit(S[train],y[train],Z[train],s)
            loss.append(np.mean((S[valid]@wc+Z[valid]@gc-y[valid])**2))
        full=(S@w+Z@gamma-y).reshape(len(s['mu_shifts']),-1)
        print(''.join(p), 'objective',round(obj,9),'shift_RMSE',np.round(np.sqrt(np.mean(full**2,axis=1)),6),'LO_rosette_RMSE',round(np.sqrt(np.mean(loss)),6),'worst',round(np.sqrt(max(loss)),6),'w',np.round(w,5))
