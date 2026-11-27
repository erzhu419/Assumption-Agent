from explore import *
from scipy.optimize import least_squares
from itertools import product

def moments(g):
 lab=g['lab_id']; ref=g['ref_coupons']; return {l:(np.median(ref[lab==l].mean(0)),np.median(ref[lab==l].std(0))) for l in np.unique(lab)}
def merge(gs): return {k:np.concatenate([g[k] for g in gs]) for k in ('lab_id','ref_coupons')}
def prepare(g,panel):
 lab=g['lab_id']; x=g['coupon_features_raw'].copy(); m=moments(panel); backup=moments(g)
 for l in np.unique(lab):
  loc,sc=m.get(l,backup[l]); q=lab==l; x[q]=(x[q]-loc)/max(sc,1e-10)
 _,ix,n=np.unique(np.c_[lab,g['stratum']],axis=0,return_inverse=True,return_counts=True); w=1/n[ix]; x=np.where(np.isnan(x),np.nansum(x*w[:,None],axis=0)/np.sum(np.isfinite(x)*w[:,None],axis=0),x)
 return g['past_channel_raw']@b[1:].sum(1)+b[0].sum(),g['hold_channel_raw']@b[1:].sum(1)+b[0].sum(),x

def mat(A,ta):
 p,h,x=A; v=np.c_[np.ones(len(x)),x]; d=np.r_[np.cumsum(dt[::-1])[::-1],0.]; k=np.diff(np.exp(-d[:,None]/ta),axis=0); z=h[:,None]-p@k
 return np.concatenate((np.concatenate((z[:,None,:]*(-np.expm1(-t[:,None]/ta)),v[:,None,:]*t[None,:,None]),axis=2),np.c_[z/ta,v][:,None,:]),axis=1)

def fit2(A,y,s,frac,mask):
 a=tuple(x[mask] for x in A); y=y[mask]; best=None
 for ta in ([.25,1.5],[.65,4.],[1.4,12.]):
  x=mat(a,np.array(ta)); w=np.broadcast_to(s,y.shape); theta=np.linalg.lstsq((x/w[...,None]).reshape(-1,7),(y/w).ravel(),rcond=None)[0]
  def err(z): return ((mat(a,np.exp(z[:2]))@z[2:]-y)/s).ravel()
  v=least_squares(err,np.r_[np.log(ta),theta],bounds=(np.r_[[np.log(.05001)]*2,[-np.inf]*7],np.r_[[np.log(24.999)]*2,[np.inf]*7]),loss='soft_l1',f_scale=frac,max_nfev=140)
  if best is None or v.cost<best.cost: best=v
 z=best.x; j=np.argsort(z[:2]); return np.r_[z[:2][j],z[2:4][j],z[4:]]

tr=D['train']; pi=D['chamber_verification_trials']; te=D['test']; y=target(pi); labs=pi['lab_id']; missing=np.isnan(pi['coupon_features_raw']).any(1)
scales={'original':np.r_[np.full(len(t),.025),.045],'robust':np.r_[np.hypot(.004,.003*t),.006],'window':np.r_[np.full(len(t),.008),.004]}
for tag,panel in [('own',pi),('train',tr),('deploy',merge([pi,te])),('all',merge([tr,pi,te]))]:
 a=prepare(pi,panel)
 for scale,s in scales.items():
  er=np.zeros_like(y); pars=[]
  for lab in np.unique(labs):
   hold=labs==lab; v=fit2(a,y,s,1.,~hold); pars.append(np.exp(v[:2])); er[hold]=mat(tuple(x[hold] for x in a),np.exp(v[:2]))@v[2:]-y[hold]
  v=fit2(a,y,s,1.,np.ones(len(y),bool)); e=mat(a,np.exp(v[:2]))@v[2:]-y
  metric=lambda e: [np.mean(abs(e[:,:-1])),np.mean(abs(e[:,-1]))]
  print(tag,scale,'OOF',np.round(metric(er),5),'worst lab',np.round(np.max([metric(er[labs==l]) for l in np.unique(labs)],axis=0),5),'missing',np.round(metric(er[missing]),5),'fit',np.round(metric(e),5),'tau',np.round(np.exp(v[:2]),3),'fold tau',np.round(pars,2).tolist(),'theta',np.round(v[2:],4))
