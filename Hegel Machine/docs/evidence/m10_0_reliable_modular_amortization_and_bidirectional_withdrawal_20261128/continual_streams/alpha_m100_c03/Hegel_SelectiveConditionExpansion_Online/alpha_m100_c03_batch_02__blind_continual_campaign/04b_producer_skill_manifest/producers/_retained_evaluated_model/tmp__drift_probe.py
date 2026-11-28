from chamber_probe import *
from downstream_solver import _features
from scipy.optimize import minimize_scalar
from scipy.stats import median_abs_deviation
tr,te=d['train'],d['test']
G={k:np.concatenate([te[k],g[k]]) for k in ('coupon_features_raw','lab_id','stratum','ref_coupons')}
T=np.r_[times,0.]; n=len(tr['lab_id']); z0=z.copy(); B=basis(tau); surf=B[:,:2]@(w[:2]*gap).T
F=y-surf.T
print('drift apparent mean trajectory relative to intrinsic time, dominant two SVDs:')
u,s,v=np.linalg.svd(F,full_matrices=False); print(s); print(T); print(v[:2]); print('gap corrected rate residual by lab')
for lname,X in [('train+diag',x),('train+test+diag',_features(tr,G)[-len(h):]),('diag-only',_features(g,g)[-len(h):])]:
 rr=w[2]+X@w[3:]; er=F[:,-1]-rr; print(lname,'rate by lab',[(j,round(np.mean(er[g['lab_id']==j]),4),round(np.sqrt(np.mean(er[g['lab_id']==j]**2)),4)) for j in range(4)]); print('drift by time slope & RMSE',np.round(np.array([(np.linalg.lstsq(rr[:,None],col,rcond=None)[0][0],np.std(col-rr*np.linalg.lstsq(rr[:,None],col,rcond=None)[0][0])) for col in F.T]),5).T)
A=np.c_[ones,r,h,z,x]; out=np.linalg.lstsq(A,F,rcond=None)[0]; print('joint ls residual',stats(F-A@out)); print('rate gap corrected vs intrinsic drift'); print(np.round(np.c_[g['lab_id'],np.isnan(g['coupon_features_raw']).sum(1),r,F[:,-1],F[:,-2]],4))
for label,fun,limits in [('exp integral',lambda k: np.r_[-np.expm1(-k*times)/k,1],(.1,10)),('gaussian integral',lambda k: np.r_[np.sqrt(np.pi)/(2*k)*__import__('scipy').special.erf(k*times),1],(.1,10)),('rational quadratic',lambda k: np.r_[times/(1+k*times**2),1],(.01,10)),('exponential strain',lambda k: np.r_[times*np.exp(-k*times),1],(.01,10)),('gaussian strain',lambda k: np.r_[times*np.exp(-k*times**2),1],(.01,10)),('integrated rational',lambda k: np.r_[np.arctan(k*times)/k,1],(.01,10)),('arctanh saturation',lambda k: np.r_[np.tanh(k*times)/k,1],(.01,10)),('linear denominator',lambda k: np.r_[times/(1+k*times),1],(.01,10))]:
 def cost(k): return np.median(np.abs(F-r[:,None]*fun(k)))
 opt=minimize_scalar(cost,bounds=limits,method='bounded'); print(label,round(opt.x,5),round(opt.fun,5),np.round(stats(F-r[:,None]*fun(opt.x)),5))
print('median apparent drift shape',np.round(np.median(F[r>.2]/r[r>.2,None],axis=0),5))
