import pickle, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
from downstream_solver import _features, _loads, _design, solve_downstream
np.set_printoptions(precision=5,suppress=True)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; rawg=p['campaign_diagnostics']
d=calibrate_upstream(d); g=calibrate_upstream({**p['dataset'],'test':rawg})['test']
tr,te=d['train'],d['test']; times,dt=d['eval_times'],d['past_dt']; base=solve_downstream(d); tau,w=base['estimated_taus'],base['hereditary_weights']; print('intrinsic',tau,w)
y=np.column_stack((g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']))
past,h=_loads(g); ones=np.ones_like(h); x=_features(tr,g)[len(tr['lab_id']):]; r=w[2]+x@w[3:]; a,b=_design(tau,x,past,h,dt,times); pred=np.column_stack((a@w,b@w)); gap=b[:,:2]*tau; z=h[:,None]-gap

def stats(v):
 return [np.sqrt(np.mean(v[:,:-1]**2)),np.sqrt(np.mean(v[:,-1]**2)),np.median(np.abs(v[:,:-1])),np.median(np.abs(v[:,-1]))]

def basis(t):
 return np.column_stack((np.vstack((-np.expm1(-times[:,None]/t),1/t)),np.r_[times,1.]))

def profile(feat, target, t=tau, scale=.008, free=True, mode='linear'):
 B=basis(t); M=np.einsum('if,tk->itfk',feat,B).reshape(target.size,-1); Y=target.ravel(); p0=np.linalg.lstsq(M,Y,rcond=None)[0]; opt=least_squares(lambda p:M@p-Y,p0,jac=lambda p:M,loss='cauchy',f_scale=scale,max_nfev=100); return opt.x.reshape(-1,3),opt.fun.reshape(target.shape)

if __name__=='__main__':
 print('baseline',stats(pred-y)); print('diag y sequential range',np.min(y,0),np.max(y,0))
 for name,f in [('h,r',np.c_[h,r]),('1,h,r',np.c_[ones,h,r]),('h,z,r',np.c_[h,z,r]),('1,h,z,r',np.c_[ones,h,z,r]),('h,z,x',np.c_[h,z,ones,x]),('h,past,x',np.c_[h,past,ones,x]),('past',np.c_[h,past])]:
  c,e=profile(f,y); print(name,'fit',np.round(stats(e),5),'coeff',c.tolist())
  def obj(logt):
   c,e=profile(f,y,np.exp(logt)); return np.log1p((e/.008)**2).sum()
  from scipy.optimize import minimize
  result=minimize(obj,np.log(tau),method='Nelder-Mead',options={'xatol':1e-5,'maxiter':120}); t=np.exp(result.x); c,e=profile(f,y,t); print(' free',t,np.round(stats(e),5),'coeff',c.tolist())
 print('baseline per coupon',np.round(np.c_[h,z,r,stats(pred-y)[0]*ones,np.median(pred[:,:-1]-y[:,:-1],axis=1),pred[:,-1]-y[:,-1]],4))
