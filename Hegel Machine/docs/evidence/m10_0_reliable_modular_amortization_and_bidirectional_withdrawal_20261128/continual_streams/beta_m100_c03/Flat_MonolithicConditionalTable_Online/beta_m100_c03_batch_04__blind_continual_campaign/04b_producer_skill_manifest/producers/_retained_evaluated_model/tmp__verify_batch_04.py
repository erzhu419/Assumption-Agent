import importlib.util
import pickle
import numpy as np
from scipy.optimize import minimize

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']

def axis(d):
    k,v = s['kappa_'+d],s['vel_'+d]
    n = len(k); h = s['L'+d]/(n-1)
    a = np.zeros((n,n))
    for i in range(n):
        if i == 0 or i == n-1:
            a[i,i] = k[i]/h**2
            a[i,1 if i == 0 else n-2] = -a[i,i]
        else:
            p=(k[i]+k[i+1])/(2*h**2); q=(k[i]+k[i-1])/(2*h**2)
            a[i,i-1:i+2]=(-q-max(v[i],0)/h,p+q+abs(v[i])/h,-p-max(-v[i],0)/h)
    return a

nx,ny = s['nx'],s['ny']; ax,ay=axis('x'),axis('y')
L = np.kron(np.eye(ny),ax)+np.kron(ay,np.eye(nx))+np.diag(s['V_reaction_2d'].flatten(order='F'))
B = np.column_stack([f.flatten(order='F') for f in s['source_modes']]); ij = s['sensor_ij']; indices=ij[:,0]+nx*ij[:,1]
samples=[]; dense_residual=[]
for shift in s['mu_shifts']:
    A = L+shift*np.eye(nx*ny); U = np.linalg.solve(A,B)
    dense_residual.append(np.max(np.abs(A@U-B))); samples.append(U[indices])
S = np.vstack(samples)
np.testing.assert_allclose(m._responses(s)[0],S,rtol=1e-11,atol=1e-12)

sw=s['sensor_calibration_sweeps']; x=sw['ref_rosette']; raw=sw['raw_rosette']
e = np.empty_like(x)
for i in range(5):
    mask=np.arange(len(x))%5==i
    cal=dict(sw,ref_rosette=x[~mask],raw_rosette=raw[~mask]); fit=dict(s,sensor_calibration_sweeps=cal)
    e[mask]=m._calibrate(fit,raw[mask]).reshape(-1,4)-x[mask]
r=np.max(abs(e),axis=1); clean=r<1e-3
print('OOF calibration bank maxima on clean rows',np.max(abs(e[clean]),axis=0),'gross rows',np.flatnonzero(~clean).tolist())
assert clean.sum()>=0.9*len(x) and np.max(abs(e[clean]))<1e-4

out = m.solve_batch(s); w=out['weights']; c=out['kkt_certificate']; nu=c['nu_sum']; lc=np.asarray(c['lam_lower']); uc=np.asarray(c['lam_upper'])
np.testing.assert_array_equal(c['weights'],w)
field = (U@w).reshape(nx,ny,order='F')
np.testing.assert_allclose(out['terminal_field'],field,rtol=1e-11,atol=1e-12)
assert np.max(abs(A@field.flatten(order='F')-B@w))<1e-10
Z=s['Z_nuisance']; P=np.eye(len(S))-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T); X=P@S; y=P@m._calibrate(s,s['y_obs_raw']); cap=s['u_box']; budget=s['total_budget']; ridge=s['source_ridge']
def objective(w):
    d=X@w-y; return (d@d+ridge*(w@w))/2

def grad(w): return X.T@(X@w-y)+ridge*w

ref=minimize(objective,np.full(len(w),budget/len(w)),jac=grad,bounds=[(0,cap)]*len(w),constraints={'type':'eq','fun':lambda w:w.sum()-budget,'jac':lambda w:np.ones(len(w))},method='SLSQP',options={'ftol':1e-14,'maxiter':200})
assert ref.success and abs(objective(w)-ref.fun)<1e-11
assert w.min()>=-1e-12 and w.max()<=cap+1e-12 and abs(w.sum()-budget)<1e-12
assert min(lc.min(),uc.min())>=0
kkt=max(np.max(abs(grad(w)-nu-lc+uc)),np.max(abs(w*lc)),np.max(abs((cap-w)*uc)))
independent_dual=objective(w)-grad(w)@w+nu*budget+cap*np.minimum(0,grad(w)-nu).sum()
gap=c['primal_objective']-c['dual_lower_bound']
assert kkt<1e-10 and -1e-11<=gap<1e-10
assert abs(independent_dual-c['dual_lower_bound'])<1e-11 and abs(objective(w)-c['primal_objective'])<1e-11
print('max shifted dense PDE residual',max(dense_residual),'terminal Poisson residual',np.max(abs(A@field.flatten(order='F')-B@w)),'KKT infinity residual',kkt,'primal-dual gap',gap,'independent objective delta',objective(w)-ref.fun)
