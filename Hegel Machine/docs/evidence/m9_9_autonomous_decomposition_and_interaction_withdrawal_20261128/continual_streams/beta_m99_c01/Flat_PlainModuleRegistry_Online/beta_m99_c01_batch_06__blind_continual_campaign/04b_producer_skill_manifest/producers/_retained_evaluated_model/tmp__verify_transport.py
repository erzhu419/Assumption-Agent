import sys, pickle, runpy
sys.path.insert(0, '/holdout_task')
import numpy as np
from batch_solver import solve_batch
from _verified_upstream import calibrate_upstream
from scipy.optimize import minimize
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s=calibrate_upstream(item['study']); nx,ny=s['nx'],s['ny']; n=nx*ny
ix,iy=s['sensor_ij'].T; modes=s['source_modes']; z=s['Z_nuisance']; y=s['y_obs_raw']; cap,budget=s['u_box'],s['total_budget']; lam=s['source_ridge']

def reference(walls):
    A=np.diag(s['V_reaction_2d'].flatten('F'))
    for j in range(ny):
        for i in range(nx):
            row=i+nx*j
            for d,idx,size,stride,wall in zip('xy',(i,j),(nx,ny),(1,nx),walls):
                k=s['kappa_'+d]; v=s['vel_'+d][idx]; h=s['L'+d]/(size-1)
                if wall=='R' and idx in (0,size-1):
                    scale=k[idx]/h**2; A[row,row]+=scale; A[row,row+(1 if idx==0 else -1)*stride]-=scale
                else:
                    for sign in (-1,1):
                        nb=(idx+sign)%size; scale=(k[idx]+k[nb])/(2*h**2)+max(-sign*v,0)/h
                        A[row,row]+=scale; A[row,row+(nb-idx)*stride]-=scale
    B=np.column_stack([m.flatten('F') for m in modes]); fields=[np.linalg.solve(A+mu*np.eye(n),B) for mu in s['mu_shifts']]
    return np.vstack([f[ix+nx*iy] for f in fields]),fields[-1]

def profile(S,mask):
    D=np.column_stack((S,y)); Z=z[mask]; P=np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(z.shape[1]),Z.T@D[mask]); T=D[mask]-Z@P
    X,t=T[:,:-1],T[:,-1]
    def fg(w):
        r=X@w-t
        return .5*(r@r+lam*(w@w)),X.T@r+lam*w
    return fg,P

out=solve_batch(item['study']); w=out['weights']; c=out['kkt_certificate']; full=np.ones(len(y),bool)
for walls in ('PP','PR'):
    S,F=reference(walls); errors=[]
    for shift in range(len(s['mu_shifts'])):
        test=np.arange(len(y))//len(ix)==shift
        fg,P=profile(S,~test)
        fit=minimize(fg,np.full(len(w),budget/len(w)),jac=True,method='SLSQP',bounds=[(0,cap)]*len(w),constraints={'type':'eq','fun':lambda u:u.sum()-budget,'jac':lambda u:np.ones_like(u)},options={'ftol':1e-14}); assert fit.success
        u=fit.x; e=S[test]@u+z[test]@(P[:,-1]-P[:,:-1]@u)-y[test]; errors.append(np.sqrt(np.mean(e**2)))
    print(walls,'leave-shift-out RMSE',np.round(errors,7))
    if walls!='PP': continue
    fg,_=profile(S,full); obj,g=fg(w); nu=c['nu_sum']; lo=np.asarray(c['lam_lower']); hi=np.asarray(c['lam_upper']); lb=obj-g@w+nu*budget+cap*np.minimum(0,g-nu).sum()
    checks={'budget':abs(w.sum()-budget),'box':max(0,-w.min(),w.max()-cap),'stationarity':max(abs(g-nu-lo+hi)),'complementarity':max(max(abs(lo*w)),max(abs(hi*(cap-w)))),'field':max(abs((F@w).reshape(nx,ny,order='F')-out['terminal_field']).flatten()),'objective':abs(c['primal_objective']-obj),'lower_bound':abs(c['dual_lower_bound']-lb),'duality_gap':abs(obj-lb),'certificate_weights':max(abs(np.asarray(c['weights'])-w))}; print(checks); assert max(checks.values())<1e-10
runpy.run_path('/holdout_task/validate_public_schema.py',run_name='__main__')
