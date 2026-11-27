exec(open('/tmp/chamber_probe.py').read().split('for label,A in [')[0])
from scipy.optimize import least_squares
keys=('coupon_features_raw','ref_coupons','lab_id','stratum')
# Use the deployment reference panel to align diagnostic rows without changing the training fit.
joined={k:np.concatenate((tr[k],te[k])) for k in keys}
_,Xall=archive._standardize(joined,g)
Xbase=g['coupon_features_raw'].copy()
for lab in np.unique(g['lab_id']):
 sel=joined['lab_id']==lab; c=joined['ref_coupons'][sel]; mask=g['lab_id']==lab
 Xbase[mask]=(Xbase[mask]-np.nanmedian(np.nanmean(c,0)))/np.nanmedian(np.nanstd(c,0))
Xbase=np.where(np.isfinite(Xbase),Xbase,Xall)
Z=z(tau); n=len(Y); target=np.c_[Y,R]; lab=g['lab_id']; ids=np.arange(n)

def chamber(p,D,mode):
 if mode=='full':
  q=tau*np.exp(p[:2]); gain=p[2:4]; carry=p[4:6]; a,b,ld,c=p[6:]
 elif mode=='reset':
  q=tau*np.exp(p[:2]); gain=p[2:4]; carry=[0.,p[4]]; a,b,ld,c=p[5:]
 elif mode=='equilibration':
  q=tau*np.exp(p[:2]); gain=p[2:4]; carry=np.exp(-np.exp(p[4])/tau); a,b,ld,c=p[5:]
 elif mode=='unity':
  q=tau*np.exp(p[:2]); gain=[1.,1.]; carry=np.exp(-np.exp(p[2])/tau); a,b,ld,c=p[3:]
 elif mode=='intrinsic':
  q=tau; gain=p[:2]; carry=np.exp(-np.exp(p[2])/tau); a,b,ld,c=p[3:]
 E=th[:2]*(H[:,None]*gain-Z*carry); dur=np.exp(ld); F=-dur*np.expm1(-t/dur)
 return np.c_[E@(-np.expm1(-t[:,None]/q)).T+D[:,None]*(a*t+b*F)+c*t,E@(1/q)+D*(a+b)+c]
starts={'full':[0,-.3,.94,1,0,.27], 'reset':[0,-.3,.94,1,.27], 'equilibration':[0,-.3,.94,1,1.6], 'unity':[0,-.3,1.6], 'intrinsic':[.94,1,1.6]}
for label,DD in [('all',np.c_[np.ones(n),Xall]@th[2:]),('base',np.c_[np.ones(n),Xbase]@th[2:])]:
 for mode,start in starts.items():
  start=np.r_[start,.23,.79,-.05,.005]
  def fit(keep):
   def res(p): return (chamber(p,DD,mode)[keep]-target[keep]).ravel()
   return least_squares(res,start,loss='cauchy',f_scale=.009,max_nfev=350).x
  p=fit(ids); e=chamber(p,DD,mode)-target; e_cv=np.zeros_like(e)
  for k in np.unique(lab):
   keep=lab!=k; pp=fit(keep); e_cv[~keep]=chamber(pp,DD,mode)[~keep]-target[~keep]
  worst=[max(np.sqrt(np.mean(e_cv[lab==k][:,:-1]**2)) for k in np.unique(lab)),max(np.sqrt(np.mean(e_cv[lab==k][:,-1]**2)) for k in np.unique(lab))]
  print(label,mode,'p',np.round(p,5),'typical',np.round(np.median(abs(e_cv),0)[[0,5,11,12]],5),'worst lab',np.round(worst,5),'pooled LOLO',np.round([np.sqrt(np.mean(e_cv[:,:-1]**2)),np.sqrt(np.mean(e_cv[:,-1]**2))],5),'train ranked',np.round(np.sort(np.sqrt(np.mean(e**2,1)))[-5:],5))
