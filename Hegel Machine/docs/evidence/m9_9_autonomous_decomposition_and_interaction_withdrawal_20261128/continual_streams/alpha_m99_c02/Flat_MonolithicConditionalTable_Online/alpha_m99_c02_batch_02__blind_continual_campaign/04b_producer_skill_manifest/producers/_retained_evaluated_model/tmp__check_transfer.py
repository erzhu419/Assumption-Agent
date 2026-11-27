exec(open('/tmp/explore_chamber.py').read().split("x=diag_features([g]);")[0])
from itertools import product
past,hold=m._loads(g,b)
tb=np.vstack([np.c_[-np.expm1(-t[:,None]/taus),t],np.r_[1/taus,1]])
y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]
lab=g['lab_id']; miss=np.isnan(g['coupon_features_raw']).any(1)
for refname,xf in [('diag',diag_features([g])),('train+test',diag_features([tr,te])),('all',diag_features([tr,te,g]))]:
 curve,rate=m._basis(taus,past,hold,xf,dt,t); base=np.c_[curve@theta,rate@theta]; drift=theta[2]+xf@theta[3:]; z=hold[:,None]-rate[:,:2]*taus
 print(refname,'baseline-rate abs',np.round(abs(base[:,-1]-y[:,-1]),4),'missing',np.where(miss)[0])
 for name,A in [('drift-free',drift[:,None,None]*tb),('initial-rate-conserving',drift[:,None,None]*(tb[:,:2]-tb[:,2:]/taus)),('plus-memory',np.concatenate([drift[:,None,None]*tb,z[:,None,:]*tb[None,:,:2]],axis=2))]:
  err=np.empty_like(y); params=[]
  for l in [None,*np.unique(lab)]:
   fit=np.ones(len(y),bool) if l is None else lab!=l
   a=A[fit].reshape(-1,A.shape[-1]); target=(y-base)[fit].ravel(); p=least_squares(lambda p:a@p-target,np.linalg.lstsq(a,target,rcond=None)[0],loss='soft_l1',f_scale=.004).x
   if l is None: print(name,'coef',p.round(5),'all fit curve/rate',np.mean(abs(A@p+base-y),axis=0)[[0,-1]].round(5))
   else: err[~fit]=(A@p+base-y)[~fit];params.append(p)
  print('  LOLO curve/rate per lab',[(round(abs(err[lab==l,:-1]).mean(),5),round(abs(err[lab==l,-1]).mean(),5)) for l in np.unique(lab)],'coef spread',np.std(params,axis=0).round(4))
