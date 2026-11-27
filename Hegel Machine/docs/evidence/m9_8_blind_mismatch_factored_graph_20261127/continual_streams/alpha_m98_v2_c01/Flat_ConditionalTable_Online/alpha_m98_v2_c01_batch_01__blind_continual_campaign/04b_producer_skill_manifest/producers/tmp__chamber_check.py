import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from probe import *
from scipy.optimize import least_squares
sp=m._specimen(c,m._features(c,c)[1],bm); p,h,X=sp
D=X@th[2:]; Y=np.column_stack([c['strain_curves_obs']-c['strain_0'][:,None],c['instant_rate_obs']]); k=len(t)
def basis(tau): return np.vstack([-np.expm1(-t[:,None]/tau),1/tau])
B=basis(tau); base=(h[:,None]*th[:2])@B.T
print('fit fixed taus and hold weights', np.linalg.lstsq((D[:,None,None]*B).reshape(-1,2),(Y-base).ravel(),rcond=None)[0]); print('diag feature scales')
for l in np.unique(c['lab_id']):
 print(l,[(round(np.median(g['ref_coupons'][g['lab_id']==l].mean(0)),4),round(np.median(g['ref_coupons'][g['lab_id']==l].std(0)),4)) for g in (c,tr,te)])
forms={'coupled':lambda q,j: (h[j,None]*th[:2]+D[j,None]*q)@B.T,'freehold':lambda q,j: np.column_stack([h[j],D[j]])@q.reshape(2,2)@B.T,'history':lambda q,j: (h[j,None]*th[:2]+D[j,None]*q[:2]+(h[j,None]-m._branches(sp,dt,t,tau)[1][j]*tau)*q[2:])@B.T,'lineardrift':lambda q,j: (h[j,None]*th[:2]+D[j,None]*q[:2])@B.T + D[j,None]*q[2]*np.r_[t,1], 'extraoffset':lambda q,j: (h[j,None]*th[:2]+D[j,None]*q[:2]+q[2:])@B.T}
for name,f in forms.items():
 size={'coupled':2,'lineardrift':3}.get(name,4)
 fit=lambda j:least_squares(lambda q:(f(q,j)-Y[j]).ravel(),np.zeros(size)).x
 allj=np.arange(len(h)); q=fit(allj); err=np.zeros_like(Y)
 for lab in np.unique(c['lab_id']):
  j=c['lab_id']!=lab; err[~j]=f(fit(j),~j)-Y[~j]
 print(name,np.round(q,6),'pooled',np.round(np.sqrt(np.mean(err**2,axis=0)),5),'worst lab curve/rate',np.max([np.sqrt(np.mean(err[c['lab_id']==l,:k]**2)) for l in np.unique(c['lab_id'])]),np.max([np.sqrt(np.mean(err[c['lab_id']==l,k]**2)) for l in np.unique(c['lab_id'])]))
print('two branch joint diagnostic inference (development only)')
q=least_squares(lambda q: ((np.c_[h,D]@q[2:].reshape(2,2))@basis(np.exp(q[:2])).T-Y).ravel(),np.r_[np.log(tau),th[:2],.35,2.15]).x
print(np.exp(q[:2]),q[2:]); print('test hold/drift range',[(float(np.min(z)),float(np.max(z))) for z in (m._specimen(te,y,bm)[1],th[2]+y@th[3:])]); print('diag hold/drift range',[(float(z.min()),float(z.max())) for z in (h,D)])
