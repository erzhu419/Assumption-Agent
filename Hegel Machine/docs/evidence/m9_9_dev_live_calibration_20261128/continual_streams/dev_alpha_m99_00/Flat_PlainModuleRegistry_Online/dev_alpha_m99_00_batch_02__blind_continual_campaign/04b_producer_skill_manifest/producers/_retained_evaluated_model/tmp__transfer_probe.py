from probe import *
from scipy.optimize import least_squares
labs=cd['lab_id']; dl=cd['lab_id']; raw=cd['coupon_features_raw']; finite=np.isfinite(raw).all(1)
for lab in np.unique(dl):
 print('panels',lab,[(len(s['lab_id'][s['lab_id']==lab]),np.median(s['ref_coupons'][s['lab_id']==lab].mean(0)),np.median(s['ref_coupons'][s['lab_id']==lab].std(0))) for s in [c['train'],c['test'],cd]])

def align(s, panels, fill):
 a=s['coupon_features_raw'].copy()
 for l in np.unique(s['lab_id']):
  v=np.concatenate([p['ref_coupons'][p['lab_id']==l] for p in panels]); m=s['lab_id']==l
  a[m]=(a[m]-np.median(v.mean(0)))/np.median(v.std(0))
 return np.where(np.isfinite(a),a,fill)
for spec in ['target','three','train','diagnostic','mixed']:
 ss={'target':(c['train'],c['test']), 'three':(c['train'],c['test'],cd),'train':(c['train'],),'diagnostic':(c['train'],cd),'mixed':(c['train'],c['test'])}[spec]; x0=prior._features(*ss); tc=prior._coupon(c['train'],x0[:n],br); a,b=prior._fit(tc,dt,t,y,r)
 dd=align(cd, [cd] if spec=='mixed' else ss,np.nanmean(x0,0)); dc=prior._coupon(cd,dd,br); A,C=prior._design(a,dc,dt,t); B=A[:,:,:2]@b[:2]; I=C[:,:2]@b[:2]; D=C[:,2:]@b[2:]; E=Y-B
 def terms(v,model):
  k=v[0]; f=-np.expm1(-k*t)/k; amp=D
  if model=='exp+lin': f=(1-v[1])*f+v[1]*t
  if model=='quadratic': amp=D+v[1]*D**2
  if model=='slope': amp=D*v[1]
  if model=='offset': amp=D+v[1]
  if model=='feature': amp=D+C[:,2:]@v[1:]
  return f,amp
 for model in ['exp','exp+lin','quadratic','slope','offset','feature']:
  init=[1.]+({'exp':[],'slope':[1.],'feature':[0.]*5}.get(model,[0.]))
  def fit(m):
   def fun(v):
    f,amp=terms(v,model); return np.r_[(E[m]-amp[m,None]*f).ravel(),(R[m]-I[m]-amp[m])]
   return least_squares(fun,init,loss='soft_l1',f_scale=.009,bounds=([.05]+[-np.inf]*(len(init)-1),[10]+[np.inf]*(len(init)-1))).x
  v=fit(finite); scores=[]
  for l in np.unique(labs):
   m=labs==l; ff,amp=terms(fit(~m & finite),model); err=E-amp[:,None]*ff; er=R-I-amp
   scores.append([np.sqrt(np.mean(err[m]**2)),np.sqrt(np.mean(err[m,-1]**2)),np.sqrt(np.mean(err[m&finite]**2)),np.sqrt(np.mean(er[m]**2))])
  print(spec,model,np.round(v,4),np.round(np.max(scores,0),5))
 f,amp=terms(fit(finite),'exp'); q=E@f/(f@f); print('amp error by lab, complete',[(l,np.round(np.mean((q-D)[(dl==l)&finite]),4)) for l in np.unique(dl)]); print('train tau',a)
