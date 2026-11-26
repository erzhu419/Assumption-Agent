import pickle, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
import alpha_w04_batch_01_solver as s
c=pickle.load(open('/holdout_task/public_alpha_w04_batch_01.pkl','rb'))[0]; d,g=c['dataset'],c['gt']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; cal=s._calibrate(d['calibration_load_cells']); x,xt=s._align(tr,te); h,u=s._stress(tr,cal); ht,ut=s._stress(te,cal); z,zt=tr['initial_memory_states'],te['initial_memory_states']; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; miss=np.isnan(tr['coupon_features_raw']).any(axis=1); mt=np.isnan(te['coupon_features_raw']).any(axis=1); lab=tr['lab_id']; lt=te['lab_id']
rmse=lambda e:np.sqrt(np.mean(e**2))
rng=np.random.default_rng(544); folds=np.empty(len(y),int)
for l in range(4):
 for missing in (0,1):
  ii=np.flatnonzero((lab==l)&(miss==missing)); folds[rng.permutation(ii)]=np.arange(len(ii))%4

def run(fit, params_on_complete, use_test):
 chosen=fit & ~miss if params_on_complete else fit
 tau,w=s._fit(dt,t,h[chosen],u[chosen],x[chosen],z[chosen],y[chosen],r[chosen]); a,b=s._design(tau,dt,t,h,u,x,z); base=a@w; drift=b@w
 # Each late strain reading measures linear drift after subtracting the hereditary response.
 slope=np.median((y-base)[:,t>=1]/t[t>=1],axis=1); score=x@w[3:]
 p=np.zeros((4,2)); variance=np.zeros(4); mad=lambda q:1.4826*np.median(abs(q-np.median(q)))
 for l in range(4):
  idx=fit&~miss&(lab==l)
  if idx.sum()>=8:
   mat=np.column_stack([np.ones(idx.sum()),score[idx]])
   p[l]=least_squares(lambda q:mat@q-slope[idx],np.zeros(2),loss='soft_l1',f_scale=.004).x
   variance[l]=mad(slope[idx]-mat@p[l])**2
  else:
   variance[l]=mad(slope[fit&~miss])**2
 if use_test:
  a,b=s._design(tau,dt,t,ht,ut,xt,zt); base=a@w; drift=b@w; R=te['instant_rate_obs']; M=mt; L=lt; S=xt@w[3:]; Y=g['te_clean']-te['strain_0'][:,None]; H=ut; i0=te['strain_0']; critical=te['critical_strain']; Cross=g['te_cross']
 else:
  j=~fit; base=base[j]; drift=drift[j]; R=r[j]; M=miss[j]; L=lab[j]; S=score[j]; Y=y[j]; H=u[j]
 corr=R-drift; offset=p[L,0]+S*p[L,1]; alpha=variance[L]/(variance[L]+max(.003,mad((r-bt(tau,w))[fit&~miss]))**2); alpha=np.minimum(alpha,.65)
 methods=[np.zeros_like(corr),np.where(M,corr,.25*corr),np.where(M,corr,offset),np.where(M,corr,offset+.25*(corr-offset)),np.where(M,corr,offset+alpha*(corr-offset))]
 pred=np.array([base+q[:,None]*t for q in methods]); e=pred-Y
 if use_test:
  print('complete-only fit',params_on_complete,'taus',tau,'lab corrections',p.round(5).tolist(),'noise estimates',np.sqrt(variance).round(5),'rate fusion',np.round([alpha[L==l][0] for l in range(4)],3))
  for l in range(4):
   j=L==l
   print(l,'curve',np.sqrt(np.mean(e[:,j]**2,axis=(1,2))).round(5),'secant',np.sqrt(np.mean((e[:,j,-1]/np.maximum(abs(H[j]),.5))**2,axis=1)).round(5),'cross',np.array([rmse(s._crossing(t,A+i0[:,None],i0,critical)[j]-Cross[j]) for A in pred]).round(5))
 return L,M,e

def bt(tau,w):
 return s._design(tau,dt,t,h,u,x,z)[1]@w
for opt in (False,True):
 for protocol in ('within','lab'):
  results=[run((folds if protocol=='within' else lab)!=i,opt,False) for i in range(4)]
  L=np.concatenate([q[0] for q in results]); M=np.concatenate([q[1] for q in results]); E=np.concatenate([q[2] for q in results],axis=1)[:,:,-1]
  print(opt,protocol,'[baseline,shared fusion,lab adjustment,lab fusion,variance fusion]')
  for l in range(4):
   for m in (False,True):
    ii=(L==l)&(M==m); err=E[:,ii]; print(l,m,ii.sum(),'rmse',np.sqrt(np.mean(err**2,axis=1)).round(4),'medAE',np.median(abs(err),axis=1).round(4))
 run(np.ones(len(y),bool),opt,True)
