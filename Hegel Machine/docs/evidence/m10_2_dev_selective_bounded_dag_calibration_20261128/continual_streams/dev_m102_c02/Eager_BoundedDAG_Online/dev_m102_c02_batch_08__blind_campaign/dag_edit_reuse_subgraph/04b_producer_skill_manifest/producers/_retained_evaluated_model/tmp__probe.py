import numpy as np,pickle,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; rng=np.random.default_rng(331)
for k in ('panel_a','panel_b','panel_c'):
 p=s[k]; u=p['u']; n,d=u.shape
 print(k)
 for vk,v in p.items():
  if vk=='u':continue
  models={'lin':np.c_[np.ones(n),u], 'diagquad':np.c_[np.ones(n),u,u*u], 'cyclic':np.c_[np.ones(n),u,u*np.roll(u,-1,axis=1)],'quad':np.c_[np.ones(n),u,*[u[:,i]*u[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)]], 'cubic':np.c_[np.ones(n),u,u**3], 'tanh':np.c_[np.ones(n),np.tanh(u)],'atan':np.c_[np.ones(n),np.arctan(u)]}
  for name,a in models.items():
   h=len(a.T); b=np.linalg.lstsq(a,v,rcond=None)[0]; r=v-a@b
   loo=r/(1-np.sum(a*np.linalg.pinv(a).T,axis=1))[:,None]
   print(vk,name,'OLS LOO',np.mean(abs(loo)).round(5),end='; ')
   if k=='panel_a' and h<=15:
    ids=np.argsort(rng.random((1800,n)),axis=1)[:,:h+1]; bs=np.linalg.pinv(a[ids])@v[ids]; rs=np.abs(a@bs-v); err=np.mean(np.partition(rs,34,axis=1)[:,:35]**2,axis=1); best=bs[np.argmin(err,axis=0),:,np.arange(d)].T
    for _ in range(7):
     re=np.abs(a@best-v); ix=np.argsort(re,axis=0)[:39]; best=np.stack([np.linalg.lstsq(a[ix[:,j]],v[ix[:,j],j],rcond=None)[0] for j in range(d)],axis=1)
    re=np.abs(a@best-v); print('resids',np.quantile(re,[.5,.8,.9,1],axis=0).round(4));
    if name in ('lin','diagquad','cyclic','quad'): print(best.round(4)); print('row outliers',np.where(re.max(1)>.6)[0])
   else: print()
