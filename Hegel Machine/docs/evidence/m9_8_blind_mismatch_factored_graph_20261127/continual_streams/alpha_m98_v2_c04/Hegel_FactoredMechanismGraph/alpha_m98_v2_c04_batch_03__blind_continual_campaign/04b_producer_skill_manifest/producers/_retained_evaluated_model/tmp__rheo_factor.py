from rheo_probe import *
from scipy.optimize import least_squares
L=g['lab_id']; D=th[2]+xg@th[3:]; E=-np.expm1(-t[:,None]/tau); M=pg@(-np.expm1(-dt[:,None]/tau)*np.exp(-np.r_[np.cumsum(dt[:0:-1])[::-1],0][:,None]/tau)); G=(hg[:,None]-M)*th[:2]; Y=np.c_[yc,yr]; H=np.vstack((E,1/tau)); B=Y-G@H.T
print('t',t,'dt',dt); print('B singular values',np.linalg.svd(B,compute_uv=False)); print('curve-only singular values',np.linalg.svd(B[:,:-1],compute_uv=False)); print('OLS coefficient time profiles for intercept,d,X separately:'); print(np.round(np.linalg.lstsq(np.c_[np.ones(len(D)),D],B,rcond=None)[0],4)); print(np.round(np.linalg.lstsq(np.c_[np.ones(len(D)),xg],B,rcond=None)[0],4)); print('missing by row',np.isnan(g['coupon_features_raw']).sum(1)); print('diagnostic ref vs train/test by lab');
for l in np.unique(L):
 q=ref[lab==l]; z=g['ref_coupons'][L==l]; print(l,len(z),np.round([np.median(q.mean(0)),np.median(q.std(0)),np.median(z.mean(0)),np.median(z.std(0))],4))
print('separable temporal families, additional saturating offset, held-lab diagnostics')
for kind in ['sat','rat','rat2','atan','stretch','log','tanh','expquad']:
 def temporal(k):
  u=k*t
  v={'sat':-np.expm1(-u)/k,'rat':t/(1+u),'rat2':t/(1+u*u),'atan':np.arctan(u)/k,'stretch':t*np.exp(-u),'log':np.log1p(u)/k,'tanh':np.tanh(u)/k,'expquad':t*np.exp(-u*u)}[kind]; return np.r_[v,1.]
 for mode in ['affine','offset','hold_offset','vary_memory']:
  def pred(p):
   f=temporal(np.exp(p[0])); C=(p[1]*D+p[2])[:,None]*f
   if mode in ('offset','hold_offset'): C+=((p[3]+(p[5]*(hg-hg.mean()) if mode=='hold_offset' else 0)))*np.r_[-np.expm1(-t*np.exp(p[4])),np.exp(p[4])] if mode=='offset' else (p[3]+p[5]*(hg-hg.mean()))[:,None]*np.r_[-np.expm1(-t*np.exp(p[4])),np.exp(p[4])]
   if mode=='vary_memory': C+= (M*p[3:5])@H.T
   return C
  init=[-1,1,0]+{'affine':[],'offset':[-.05,0],'hold_offset':[-.05,0,0],'vary_memory':[0,0]}[mode]; scores=[]
  for l in [None,*np.unique(L)]:
   m=np.ones(len(L),bool) if l is None else L!=l
   pp=least_squares(lambda p:((pred(p)-B)[m]*np.r_[np.ones(len(t)),.65]).ravel(),init,loss='soft_l1',f_scale=.007,max_nfev=150,bounds=([-5,-5,-5]+[-5]*(len(init)-3),[5,5,5]+[5]*(len(init)-3))).x; r=(pred(pp)-B)[m if l is None else ~m]; sc=[np.sqrt(np.mean(r[:,:-1]**2)),np.sqrt(np.mean(r[:,-1]**2))]
   if l is None: full=sc; par=pp
   else: scores.append(sc)
  print(kind,mode,np.round(par,3),np.round(full,4),np.round(np.max(scores,0),4))
