import pickle,itertools as it,numpy as np
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))
if isinstance(p,list): p=p[0]
p=p["study"]
print('keys',list(p))
np.set_printoptions(precision=5,suppress=True)
for k,d in p.items():
 if not isinstance(d,dict): print(k,np.asarray(d));continue
 print(k,[(a,np.shape(b)) for a,b in d.items()]);print({a:b[:4] for a,b in d.items() if isinstance(b,np.ndarray) and b.ndim>0})
 if not k.startswith('panel'): continue
 u=d['u']; v=d.get('v',d.get('v1')); x=np.c_[np.ones(len(u)),u]; q=np.column_stack([u[:,i]*u[:,j] for i,j in it.combinations_with_replacement(range(u.shape[1]),2)])
 for name,A in [('affine',x),('quadratic',np.c_[x,q]),('harmonic',np.c_[x,np.sin(u),np.cos(u)]),('both',np.c_[x,q,np.sin(u),np.cos(u)])]:
  c=np.linalg.lstsq(A,v,rcond=None)[0]; r=v-A@c; hat=np.sum(np.linalg.svd(A,full_matrices=False)[0][:,:np.linalg.matrix_rank(A)]**2,axis=1)
  print(name,'rank',np.linalg.matrix_rank(A),'rmse',np.sqrt(np.mean(r*r,axis=0)),'loo',np.mean(abs(r/(1-hat[:,None])),axis=0) if A.shape[1]<len(u) else None); print('coefs',c)
  if name=='affine': print('row errors',np.linalg.norm(r,axis=1))
