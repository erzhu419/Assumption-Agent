import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from m98_factor import *

def features(ix,imp):
 V=dg['coupon_features_raw'].copy(); st=dg['stratum']; R=dg['ref_coupons'];
 for l in np.unique(labs):
  m=labs==l; z=R[m]; V[m]=(V[m]-np.median(z.mean(0)))/np.median(z.std(0))
 _,j,n=np.unique(np.c_[labs[ix],st[ix]],axis=0,return_inverse=True,return_counts=True); w=1/n[j].reshape(-1,1); mu=np.nansum(w*V[ix],axis=0)/np.sum(w*~np.isnan(V[ix]),axis=0)
 return np.where(np.isnan(V),mu if imp=='diag' else D,V)

def fit(ix,mem,amp,imp):
 r=th[2]+features(ix,imp)@th[3:];
 def predict(p):
  lam=np.exp(p[0]); dk=np.r_[-np.expm1(-t*lam),lam]; v=p[1:];
  if mem=='clock': Z=zstate(P,dt,taus*np.exp(v[:2])); j=2
  if mem=='tail': Z=zstate(P[:,-3:],dt[-3:],taus*np.exp(v[:2])); j=2
  if mem=='clock-age':
   dur=dt*np.exp(v[:6]); Z=zstate(P,dur,taus); j=6
  if mem=='gate':
   gate=np.r_[np.full(3,v[0]),np.ones(3)]; Z=zstate(P*gate,dt,taus*np.exp(v[1:3])); j=3
  if mem=='tail-fixed': Z=zstate(P[:,-3:],dt[-3:],taus); j=0
  if mem=='fixed': Z=zstate(P,dt,taus); j=0
  if amp=='scale': rd=v[j]*r
  elif amp=='affine':rd=v[j]*r+v[j+1]
  elif amp=='fixed':rd=r
  return (H[:,None]-Z)*th[:2]@E.T+rd[:,None]*dk
 j={'clock':2,'tail':2,'clock-age':6,'gate':3,'tail-fixed':0,'fixed':0}[mem]; a={'scale':[1.],'affine':[1.,0.],'fixed':[]}[amp]; p=np.r_[0.,np.zeros(j),a]; f=least_squares(lambda p:((predict(p)-Y)[ix]*W).ravel(),p,bounds=(-5,5),max_nfev=200);return f.x,predict(f.x)-Y
for mem in ['fixed','tail-fixed','clock','tail','clock-age','gate']:
 for amp in ['fixed','affine']:
  for imp in ['diag','train']:
   p,e=fit(ids,mem,amp,imp); cv=np.empty_like(e)
   for l in np.unique(labs):
    m=labs==l; cv[m]=fit(ids[~m],mem,amp,imp)[1][m]
   def met(e):return np.array([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,11]**2)),np.sqrt(np.mean(e[:,-1]**2))])
   print(mem,amp,imp,'param',p.round(4),'fit',met(e).round(4),'LOLO',np.max([met(cv[labs==l]) for l in np.unique(labs)],0).round(4))
