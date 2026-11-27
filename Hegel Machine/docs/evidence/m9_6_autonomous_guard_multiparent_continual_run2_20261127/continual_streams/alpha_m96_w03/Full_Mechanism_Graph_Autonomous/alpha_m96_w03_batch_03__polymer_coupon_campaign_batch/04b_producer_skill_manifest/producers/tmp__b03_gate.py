exec(open('/tmp/b03_diag.py').read().split('p=m._fit(h,tr);')[0])
from scipy.optimize import least_squares
from itertools import combinations
p=m._fit(h,tr); a,_=h.design(p[:2]); direct=[]
for j,tau in enumerate(p[:2]):
 z=h.memory[:,j].copy()
 for k,dt in enumerate(h.dt):
  decay=np.exp(-dt/tau); z=decay*z+(1-decay)*h.hist[:,k]
 direct.append((h.stress-z)[:,None]*np.r_[-np.expm1(-h.times/tau),1/tau])
err=np.max(abs(a[:,:,:2]-np.stack(direct,axis=-1))); assert err<1e-12
print('recurrence vs closed form max error',err)
inc=np.r_[h.times,1.]; raw=tr['coupon_features_raw']; complete=np.isfinite(raw).all(axis=1)
pairs=np.column_stack([x[:,i]*x[:,j] for i,j in combinations(range(x.shape[1]),2)])
triples=np.column_stack([x[:,i]*x[:,j]*x[:,k] for i,j,k in combinations(range(x.shape[1]),3)])
rat=np.column_stack([(x[:,i]-c*x[:,j])/(1+0.25*(x[:,i]+c*x[:,j])**2) for i,j in combinations(range(x.shape[1]),2) for c in (-1,1)])
lab=np.asarray(tr['lab_id']); wing=lab//2
names=['baseline','signed pairs/triples','rational pairs','stress interactions']
features=[np.empty((len(x),0)),np.c_[pairs,triples],rat,np.c_[h.stress,h.stress[:,None]*x]]
for tag,grp in [('lab',lab),('wing',wing)]:
 scores={name:[] for name in names}; biases=[]
 for label in np.unique(grp):
  train=(grp!=label); test=(grp==label)&complete
  p=m._fit(h,tr,train); a,_=h.design(p[:2]); res=y-a@p[2:]
  scale=np.maximum(np.array([m._mad(q[np.isfinite(q)]) for q in res[train&complete].T]),.006)
  good=np.isfinite(y)&(train&complete)[:,None]
  biases.append([float(np.nanmedian(res[test,-1])),float(np.nanmedian(res[test,-2]))])
  for name,f in zip(names,features):
   A=np.concatenate((a,f[:,None,:]*inc[None,:,None]),axis=-1)
   if name=='baseline': coef=p[2:]
   else:
    weighted=A/scale[None,:,None]; target=(y/scale)[good]; mat=weighted[good]; seed=np.r_[p[2:],np.zeros(f.shape[1])]
    coef=least_squares(lambda q: mat@q-target,seed,jac=lambda q:mat,loss='cauchy',f_scale=2.3849,max_nfev=80).x
   e=y[test]-A[test]@coef
   scores[name].append([np.nanmedian(abs(e[:,:-1])),np.nanmedian(abs(e[:,-2])),np.nanmedian(abs(e[:,-1]))])
 for name,values in scores.items(): print(tag,name,'worst fold medAE curve/terminal/rate',np.round(np.max(values,axis=0),6))
 print(tag,'held group signed median rate/terminal residuals',np.array(biases))
