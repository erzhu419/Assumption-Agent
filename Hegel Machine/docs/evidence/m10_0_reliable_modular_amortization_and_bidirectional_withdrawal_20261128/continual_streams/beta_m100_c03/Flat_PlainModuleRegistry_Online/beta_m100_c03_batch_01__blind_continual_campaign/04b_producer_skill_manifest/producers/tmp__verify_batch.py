import pickle, runpy
import numpy as np
from scipy.optimize import minimize

p = '/holdout_task/'
m = runpy.run_path(p + 'batch_solver.py')
item = pickle.load(open(p + 'public_batch.pkl', 'rb'))[0]
s = item['study']
out = m['solve_batch'](s)

# Independently assemble each axis by rows to catch stencil and ordering errors.
def axis(d):
    kap, vel = s['kappa_'+d], s['vel_'+d]
    n = len(kap)
    h = s['L'+d]/(n-1)
    a = np.zeros((n,n))
    for i in range(n):
        if i in (0,n-1):
            a[i,i] = kap[i]/h**2
            a[i,1 if i == 0 else n-2] = -a[i,i]
        else:
            km = (kap[i-1]+kap[i])/2
            kp = (kap[i]+kap[i+1])/2
            a[i,i-1] = -km/h**2 - max(vel[i],0)/h
            a[i,i+1] = -kp/h**2 - max(-vel[i],0)/h
            a[i,i] = (km+kp)/h**2 + abs(vel[i])/h
    return a
nx, ny = s['nx'], s['ny']
a = np.kron(np.eye(ny),axis('x')) + np.kron(axis('y'),np.eye(nx)) + np.diag(s['V_reaction_2d'].ravel(order='F'))
b = np.column_stack([mode.ravel(order='F') for mode in s['source_modes']])
fields = [np.stack([v.reshape(nx,ny,order='F') for v in np.linalg.solve(a+mu*np.eye(nx*ny),b).T]) for mu in s['mu_shifts']]
i,j = s['sensor_ij'].T
S = np.vstack([field[:,i,j].T for field in fields])
cal = s['sensor_calibration_sweeps']
ref, raw = cal['ref_rosette'], cal['raw_rosette']
C = np.linalg.lstsq(np.c_[ref,np.ones(len(ref))],raw,rcond=None)[0]
def unmix(y):
    return ((y.reshape(-1,4)-C[-1]) @ np.linalg.inv(C[:-1])).ravel()
errs = np.stack([(S@pulse['w_pulse']-unmix(pulse['y_pulse_raw'])).reshape(len(fields),-1) for pulse in item['campaign_diagnostics']])
worst = np.sqrt(np.mean(errs**2,axis=-1)).max()
assert worst < 2e-6, worst
slices = np.zeros(4)
for k in range(len(ref)):
    train = np.arange(len(ref)) != k
    Ck = np.linalg.lstsq(np.c_[ref[train],np.ones(train.sum())],raw[train],rcond=None)[0]
    slices += (np.r_[ref[k],1] @ Ck-raw[k])**2
cal_worst = np.sqrt(slices/len(ref)).max()
assert cal_worst < 2e-5, cal_worst
w = np.asarray(out['weights']); cap = s['u_box']; budget = s['total_budget']; z = s['Z_nuisance']
proj = np.eye(len(z)) - z @ np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
T, y = proj@S, proj@unmix(s['y_obs_raw']); reg = s['source_ridge']
obj = lambda v: float(.5*np.linalg.norm(T@v-y)**2+.5*reg*(v@v))
grad = lambda v: T.T@(T@v-y)+reg*v
c = out['kkt_certificate']; g = grad(w); nu = c['nu_sum']; ll, lu = np.array(c['lam_lower']),np.array(c['lam_upper'])
assert np.array_equal(w,c['weights']) and abs(w.sum()-budget)<1e-12 and min(w)>=0 and max(w)<=cap
assert min(ll)>=0 and min(lu)>=0 and np.max(np.abs(g-nu-ll+lu))<1e-12
assert max(np.max(np.abs(w*ll)),np.max(np.abs((cap-w)*lu)))<1e-12
assert abs(c['primal_objective']-obj(w))<1e-12 and abs(c['dual_lower_bound']-obj(w))<1e-12
assert np.max(np.abs(out['terminal_field']-np.einsum('k,kij->ij',w,fields[-1])))<1e-12
res = minimize(obj,np.full(len(w),budget/len(w)),jac=grad,method='SLSQP',bounds=[(0,cap)]*len(w),constraints=[{'type':'eq','fun':lambda v:v.sum()-budget,'jac':lambda v:np.ones(len(v))}],options={'ftol':1e-13})
assert res.success and abs(res.fun-obj(w))<1e-11, res.message

# Stress different active faces, including fully saturated simplex vertices.
rng=np.random.default_rng(314)
metrics=np.zeros(3)
for n in (3,8,16):
    for fraction in (0,1e-5,1/n,.49,.5,.99999,1):
        for _ in range(20):
            a=rng.normal(size=(n,n)); h=a.T@a+.04*np.eye(n); b=rng.normal(size=n)*5
            cap=.42; budget=n*cap*fraction
            v,nu=m['_box_qp'](h,b,cap,budget); delta=h@v+b-nu
            vals=[abs(v.sum()-budget),max(0,-v.min(),v.max()-cap),np.maximum(delta,0)@v+np.maximum(-delta,0)@(cap-v)]
            metrics=np.maximum(metrics,vals)
assert max(metrics)<1e-10, metrics
print('Worst pulse/shift RMSE:',worst,'Worst LOOCV channel RMSE:',cal_worst)
print('Public weights:',w,'Public certificate gap:',c['primal_objective']-c['dual_lower_bound'])
print('420 QP stress cases: worst budget, bound, complementarity errors:',metrics)
