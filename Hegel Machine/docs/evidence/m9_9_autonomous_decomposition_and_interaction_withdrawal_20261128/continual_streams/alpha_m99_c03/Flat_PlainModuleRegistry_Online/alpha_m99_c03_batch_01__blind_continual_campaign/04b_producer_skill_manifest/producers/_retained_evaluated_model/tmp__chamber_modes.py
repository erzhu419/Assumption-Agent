exec(open('/tmp/chamber_probe.py').read().split('for name,d in')[0])
from scipy.optimize import least_squares
from itertools import product
Y=dg['strain_curves_obs']-dg['strain_0'][:,None]; R=dg['instant_rate_obs']; y=np.c_[Y,R]; tt=np.r_[t,0.]; N=len(Y)
xd=m._features(tr,dg)[n:]; p,h,d=m._coupon_inputs(dg,xd,b)
print('diag ranges past/hold',p.min(),p.max(),h.min(),h.max(),'obs',np.quantile(y,[0,.5,1])); print('current feature means/std', xd.mean(0), xd.std(0))
A=lambda q: np.c_[-np.expm1(-tt[:,None]/q),tt]
def timebasis(q):
 a=A(q); a[-1]=np.r_[1/q,1.]; return a

def fit(blocks, ix=np.arange(N)):
 def mat(q):
  a=timebasis(q); return np.concatenate([f[ix,None,:]*a[None,:,j,None] for j,f in enumerate(blocks(q))],axis=-1).reshape(len(ix)*len(tt),-1)
 def res(q):
  x=mat(q); z=y[ix].ravel(); return x@np.linalg.lstsq(x,z,rcond=None)[0]-z
 def dec(v): return np.cumsum(np.exp(v))
 sol=min((least_squares(lambda v:res(dec(v)),np.log([a,c-a]),bounds=(np.log([.035,.003]),np.log([28.,40.])),max_nfev=70) for a,c in [(.35,2),(.65,4),(1.1,7),(.15,12)]),key=lambda v:v.cost)
 q=dec(sol.x); x=mat(q); v=np.linalg.lstsq(x,y[ix].ravel(),rcond=None)[0]; e=res(q).reshape(len(ix),-1); print('q',q.round(5),'curve/rate RMS',np.sqrt(np.mean(e[:,:-1]**2)).round(5),np.sqrt(np.mean(e[:,-1]**2)).round(5),'coef',[v[i:j].round(4).tolist() for i,j in zip(np.r_[0,np.cumsum([f.shape[1] for f in blocks(q)])[:-1]],np.cumsum([f.shape[1] for f in blocks(q)]))]); return q

after=np.cumsum(dt[::-1])[::-1]-dt
mem=lambda q:p@(-np.expm1(-dt[:,None]/q)*np.exp(-after[:,None]/q))
for mode in ('free','free_stress','cross_dynamic','cross_intrinsic','same_dynamic','same_intrinsic','drift_full'):
 def blocks(q):
  if mode=='free': return [np.eye(N)]*3
  if mode=='free_stress': return [np.c_[h,p,d[:,0]]]*2+[np.c_[d,h,p]]
  if mode=='drift_full': return [np.c_[h,mem(q),d]]*2+[np.c_[d,h,mem(q)]]
  z=mem(q if 'dynamic' in mode else taus); a=np.c_[h,z,d[:,0]]
  return ([a,a] if mode.startswith('cross') else [a[:,[0,1,3]],a[:,[0,2,3]]])+[d]
 print(mode,end=' '); fit(blocks)
for k in np.unique(dg['lab_id']):
 print('panel',k,[(key,round(np.median(g['ref_coupons'][g['lab_id']==k].mean(0)),3),round(np.median(g['ref_coupons'][g['lab_id']==k].std(0)),3)) for key,g in [('tr',tr),('te',te),('dg',dg)]])
