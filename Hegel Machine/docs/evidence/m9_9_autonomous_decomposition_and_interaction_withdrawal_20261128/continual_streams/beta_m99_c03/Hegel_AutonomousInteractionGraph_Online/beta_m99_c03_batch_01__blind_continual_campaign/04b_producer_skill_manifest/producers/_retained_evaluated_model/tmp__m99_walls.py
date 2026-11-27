import pickle, runpy
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.sparse.linalg import splu

item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
s = item['study']
prior = runpy.run_path('/tmp/hegel_continual_archive/beta_m99_c03_s00_solver.py')
cal = s['sensor_calibration_sweeps']
x, y = cal['ref_rosette'], cal['raw_rosette']
C = np.linalg.lstsq(np.c_[np.ones(len(x)), x], y, rcond=None)[0]
P = item['campaign_diagnostics']
Y = np.stack([np.linalg.solve(C[1:].T, (p['y_pulse_raw'].reshape(-1, 4) - C[0]).T).T.ravel() for p in P], axis=1)
nx, ny = s['nx'], s['ny']
axes = [prior['_axis_operator'](s['kappa_'+a], s['vel_'+a], s['robin_'+a], s['L'+a]).toarray() for a in 'xy']
ks = [s['kappa_'+a][[0,-1]] / (s['L'+a]/(len(s['kappa_'+a])-1))**2 for a in 'xy']
rs = [s['robin_'+a] / (s['L'+a]/(len(s['kappa_'+a])-1)) for a in 'xy']
I = sparse.eye(nx * ny, format='csc')
V = sparse.diags(s['V_reaction_2d'].ravel(order='F'), format='csc')
rhs = np.column_stack([m.ravel(order='F') for m in s['source_modes']]) @ np.column_stack([p['w_pulse'] for p in P])
i,j = s['sensor_ij'].T
sample = i + nx*j
Z = s['Z_nuisance']
Q = np.linalg.qr(Z)[0]

def predict(p):
    axis = []
    for A, k, r, params in zip(axes, ks, rs, p.reshape(2,2,3)):
        A = A.copy()
        for side, (row, inward, opposite) in enumerate(((0,1,-1),(-1,-2,0))):
            d, robin, wrap = params[side]
            A[row,row] = k[side]*(d+wrap)+r*robin
            A[row,inward] = -k[side]*d
            A[row,opposite] = -k[side]*wrap
        axis.append(sparse.csc_matrix(A))
    ax, ay = axis
    L = sparse.kron(sparse.eye(ny), ax, format='csc') + sparse.kron(ay, sparse.eye(nx), format='csc') + V
    return np.vstack([splu(L+mu*I).solve(rhs)[sample] for mu in s['mu_shifts']])

base = np.tile([1.,1.,0.],(4,1))
def model(spec, train, profile):
    labels = np.asarray(spec).reshape(4,3)
    idx = np.unique(labels[labels>=0]); start = np.array([base[labels==k][0] for k in idx]);
    def expand(z):
        p = base.copy()
        for k, v in zip(idx,z): p[labels==k] = v
        return p
    def residual(p):
        R = predict(expand(p)) - Y
        return R-Q@(Q.T@R) if profile else R
    fit = least_squares(lambda p: residual(p)[:,train].ravel(),start, bounds=(-0.05, 5), x_scale='jac', max_nfev=100, ftol=1e-11, xtol=1e-11, gtol=1e-11)
    return expand(fit.x), np.linalg.norm(residual(fit.x),axis=0)

if __name__ == '__main__':
    np.set_printoptions(precision=6,suppress=True)
    print('baseline', np.linalg.norm(predict(base)-Y,axis=0))
    patterns = {'global_robin':[[-1,0,-1]]*4,'axis_robin':[[-1,0,-1]]*2+[[-1,1,-1]]*2,'wall_robin':[[-1,i,-1] for i in range(4)],'global_inward_robin':[[0,1,-1]]*4,'axis_inward_robin':[[0,1,-1]]*2+[[2,3,-1]]*2,'wall_inward_robin':[[2*i,2*i+1,-1] for i in range(4)],'axis_robin_wrap':[[-1,0,1]]*2+[[-1,2,3]]*2,'all':np.arange(12).reshape(4,3)}
    for profile in (False,True):
        for name,spec in patterns.items():
            p,e=model(spec,[0,1,2],profile)
            loo=[model(spec,[j for j in range(3) if j!=i],profile)[1][i] for i in range(3)]
            print(profile,name,'params',p.ravel(),'resid',e,'LOPO',np.round(loo,6))
