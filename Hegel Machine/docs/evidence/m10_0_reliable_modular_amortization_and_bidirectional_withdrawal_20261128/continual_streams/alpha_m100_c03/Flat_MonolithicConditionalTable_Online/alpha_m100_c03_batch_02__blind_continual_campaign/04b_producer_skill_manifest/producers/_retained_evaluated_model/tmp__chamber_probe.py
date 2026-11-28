import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive'); import alpha_m100_c03_batch_01_solver as m
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; tr,te,di=d['train'],d['test'],i['campaign_diagnostics']; ts,dt=d['eval_times'],d['past_dt']; b=m._bridge(d['calibration_load_cells']); p,h=m._loads(tr,b); x=m._features(tr,te)[:len(h)]; tau,w=m._fit(x,p,h,dt,ts,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); p,h=m._loads(di,b); x=m._features(tr,di)[-len(h):]; g=w[2]+x@w[3:]; A,R=m._design(tau,x,p,h,dt,ts); gap=R[:,:2]*tau; y=di['strain_curves_obs']-di['strain_0'][:,None]; r=di['instant_rate_obs']; e=y-A@w; er=r-R@w; F=np.c_[np.ones(len(h)),h,g,gap,p,x];
if __name__=='__main__':
 print('time',np.round(ts,3)); print('residual rows',np.round(e[:10],3)); print('rate residual',np.round(er,3)); print('SVD',np.round(np.linalg.svd(e,compute_uv=False),3)); print('corr of [hold,drift,gaps,past,features] vs [rate residual, curve residual at times]'); print(np.round(np.corrcoef(np.c_[F[:,1:],er,e].T)[:F.shape[1]-1,F.shape[1]-1:],2));
 print('previous law diagnostic error',[(np.median(abs(a)),np.mean(a*a)**.5) for a in (lambda z:(z[0]-y,z[1]-r))(m._chamber(tau,w,x,h,ts,m._CHAMBER_LAW))]);
 for j in range(len(h)):
  k=np.c_[-np.expm1(-ts[:,None]/tau),ts]; a=np.r_[k,[np.r_[1/tau,1]]]; v=np.r_[y[j],r[j]]; q=least_squares(lambda c:a@c-v,np.linalg.lstsq(a,v,rcond=None)[0],loss='soft_l1',f_scale=.018); print(j,np.round(np.r_[gap[j],g[j],q.x,q.x-np.r_[gap[j]*w[:2],g[j]],np.median(abs(q.fun)),q.fun[-1]],4))
