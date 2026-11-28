import pickle,numpy as np,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];np.set_printoptions(precision=5,suppress=True,linewidth=140)
def aff(x):return np.column_stack([np.ones(len(x)),x])
def fit(a,b):
 w=np.linalg.lstsq(a,b,rcond=None)[0]; h=np.sum(np.linalg.pinv(a).T*a,axis=1)[:,None];r=b-a@w
 return np.sqrt(np.mean(r*r)),np.sqrt(np.mean((r/np.maximum(1-h,1e-10))**2)),w
for k,d in s.items():
 print(k,list(d) if isinstance(d,dict) else (np.shape(d),str(d)[:250]))
 if not k.startswith('panel'):continue
 for p,q in itertools.permutations(d,2):
  u,v=d[p],d[q]
  if u.ndim!=2 or v.ndim!=2:continue
  print(k,p,q,u.shape,v.shape); print('sample',u[:2],v[:2]); print('affine',fit(aff(u),v))
  for typ,g in [('sq',u*u),('cube',u**3),('tanh',np.tanh(u)),('sin',np.sin(u)),('mix2',np.column_stack([u[:,i]*u[:,j] for i in range(u.shape[1]) for j in range(i,u.shape[1])])),('pair',np.column_stack([u[:,i]*u[:,j] for i in range(u.shape[1]) for j in range(i+1,u.shape[1])]))]:
   print(typ,fit(np.column_stack([aff(u),g]),v)[:2]);
  if np.max(abs(v))<1:print('inverse tanh',fit(aff(u),np.arctanh(v)))
