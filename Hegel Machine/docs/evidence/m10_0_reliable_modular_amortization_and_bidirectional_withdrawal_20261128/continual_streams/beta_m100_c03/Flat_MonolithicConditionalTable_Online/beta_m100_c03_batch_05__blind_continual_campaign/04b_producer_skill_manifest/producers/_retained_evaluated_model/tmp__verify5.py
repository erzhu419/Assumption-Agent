import importlib.util,pickle
import numpy as np
p='/holdout_task/batch_solver.py'; spec=importlib.util.spec_from_file_location('candidate',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; nx,ny=s['nx'],s['ny']; N=nx*ny

def dense_axis(d,n):
    a=np.zeros((n,n)); h=s['L'+d]/(n-1); k=s['kappa_'+d]; v=s['vel_'+d]; robin=s['robin_'+d]
    for i in (0,n-1):
        a[i,i]=k[i]/h**2+robin/h; a[i,1 if i==0 else n-2]=-k[i]/h**2
    for i in range(1,n-1):
        km=(k[i]+k[i-1])/2; kp=(k[i]+k[i+1])/2
        a[i,i-1]=-km/h**2-max(v[i],0)/h; a[i,i+1]=-kp/h**2-max(-v[i],0)/h
        a[i,i]=(kp+km)/h**2+abs(v[i])/h
    return a
A=np.kron(np.eye(ny),dense_axis('x',nx))+np.kron(dense_axis('y',ny),np.eye(nx))+np.diag(s['V_reaction_2d'].ravel(order='F'))
X=np.column_stack([u.ravel(order='F') for u in s['source_modes']]); ii=s['sensor_ij']@np.array([1,nx]); rows=[]
for mu in s['mu_shifts']:
    field=np.linalg.solve(A+mu*np.eye(N),X); rows.append(field[ii])
S=np.vstack(rows); ss,term=m._responses(s); response_error=np.max(abs(S-ss)); assert response_error<1e-10
c=s['sensor_calibration_sweeps']; a,b=c['ref_rosette'],c['raw_rosette']; clean=np.max(abs(b),axis=1)<5; fold_error=[]; inverse_error=[]
for fold in range(4):
    held=np.arange(len(a))%4==fold; _,_,f=m._sensor_fit(s,~held); e=f(a[held&clean])-b[held&clean]; fold_error.append(np.sqrt(np.mean(e**2,axis=0)).max())
e=m._calibrate(s,b[clean].ravel()).reshape(-1,4)-a[clean]; inverse_error=np.sqrt(np.mean(e**2,axis=0)).max(); assert max(fold_error)<3e-5 and inverse_error<5e-5
y=m._calibrate(s,s['y_obs_raw']); Z=s['Z_nuisance']; ridge=s['source_ridge']; cap=s['u_box']; n=S.shape[1]
P=np.eye(len(y))-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T); D=P@S; target=P@y
worst=np.zeros(6)
for budget in (0., 1e-6, cap, 2*cap, s['total_budget'], n*cap-1e-6, n*cap):
    out=m.solve_batch(dict(s,total_budget=budget)); w=out['weights']; q=out['kkt_certificate']; nu=q['nu_sum']; lo=np.array(q['lam_lower']); hi=np.array(q['lam_upper']); g=D.T@(D@w-target)+ridge*w; obj=.5*(np.linalg.norm(D@w-target)**2+ridge*(w@w)); dual=obj-g@w+nu*budget+cap*np.minimum(0,g-nu).sum()
    assert np.all(w>=0) and np.all(w<=cap) and np.all(lo>=0) and np.all(hi>=0) and np.array_equal(w,q['weights'])
    errs=[abs(w.sum()-budget),np.max(abs(g-nu-lo+hi)),max(np.max(abs(lo*w)),np.max(abs(hi*(cap-w)))),max(abs(q['primal_objective']-obj),abs(q['dual_lower_bound']-dual)),abs(obj-dual),np.max(abs(out['terminal_field']-(field@w).reshape(nx,ny,order='F')))]; worst=np.maximum(worst,errs)
    if budget==s['total_budget']: print('public weights',w)
assert worst.max()<1e-10, worst
print('max forward response error',response_error,'worst held-out sweep RMS',max(fold_error),'worst inverse sweep RMS',inverse_error)
print('worst across budgets: mass, stationarity, complementarity, objective, duality gap, terminal field',worst)
