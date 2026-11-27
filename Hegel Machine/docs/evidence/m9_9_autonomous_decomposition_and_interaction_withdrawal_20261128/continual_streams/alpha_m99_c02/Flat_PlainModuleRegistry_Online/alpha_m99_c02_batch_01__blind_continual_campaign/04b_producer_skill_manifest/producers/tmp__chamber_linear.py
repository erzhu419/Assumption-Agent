import pickle,runpy,numpy as np
from scipy.optimize import least_squares
s=runpy.run_path('/tmp/hegel_continual_archive/alpha_m99_c02_s00_solver.py'); globals().update({k:v for k,v in s.items() if k.startswith('_') and not k.startswith('__')})
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; g=i['campaign_diagnostics']; tr=d['train']; te=d['test']; dt=d['past_dt']; t=d['eval_times']; b=_bridge(d['calibration_load_cells']); xt,xe=_features(tr,te); _,xg=_features({k:np.concatenate((tr[k],te[k])) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')},g); ta,th=_material(tr,b,xt,dt,t); p,h=_loads(g,b); cv,rt=_basis(ta,p,h,xg,dt,t); Y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]; lab=g['lab_id']; n=len(lab)
def T(q): return np.r_[np.c_[-np.expm1(-t[:,None]/q),t], [np.r_[1/q,1.]]]
def hist(q):
 z=np.zeros((n,len(q)))
 for w,a in zip(dt,p.T): z+=-np.expm1(-w/q)*(a[:,None]-z)
 return z
z=hist(ta); dr=rt[:,2:]@th[2:]; print('material:',ta,th)
for name,A in {'hold':h[:,None],'hold+history':np.c_[h,z],'hold+history+drift':np.c_[h,z,dr],'hold+past':np.c_[h,p],'hold+past+drift':np.c_[h,p,dr],'hold+past+features':np.c_[h,p,np.ones(n),xg]}.items():
 def predict(f,j,free):
  F=A[f]; B=np.linalg.lstsq(F,Y[f],rcond=None)[0]
  def calc(q):
   M=T(q); C=np.linalg.lstsq(M,B.T,rcond=None)[0].T
   return C,M
  q=ta if not free else np.exp(least_squares(lambda l:(F@calc(np.exp(l))[0]@calc(np.exp(l))[1].T-Y[f]).ravel(),np.log(ta),bounds=(np.log([.1,1.1]),np.log([1.09,40]))).x)
  C,M=calc(q); return A[j]@C@M.T,q,C
 for free in (False,True):
  e=np.zeros_like(Y)
  for l in np.unique(lab):
   j=lab==l; e[j]=predict(~j,j,free)[0]-Y[j]
  pp,q,C=predict(np.ones(n,bool),np.ones(n,bool),free); rm=lambda a:np.sqrt(np.mean(a*a)); print(name,free,'q',np.round(q,4),'train',round(rm(pp-Y),5),'LOLO curve/rate worst',np.round(np.max([[rm(e[lab==l,:-1]),rm(e[lab==l,-1])] for l in np.unique(lab)],axis=0),5)); print(np.round(C,5))
cal=d['calibration_load_cells']['single_channel_sweeps'];
for l in np.unique(cal['lab_id']):
 j=cal['lab_id']==l; print('sweep',l,np.polynomial.polynomial.polyfit(cal['cmd_stress'][j],cal['true_stress'][j],1))
