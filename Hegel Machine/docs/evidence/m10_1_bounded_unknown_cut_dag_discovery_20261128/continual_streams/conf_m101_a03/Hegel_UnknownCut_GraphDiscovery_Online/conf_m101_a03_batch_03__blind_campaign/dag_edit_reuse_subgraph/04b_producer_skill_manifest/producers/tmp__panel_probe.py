import pickle,itertools,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def report(x,y):
 h=x@np.linalg.pinv(x); r=y-h@y
 return np.round(np.sqrt(np.mean(r*r,axis=0)),7),np.round(np.max(np.abs(r/(1-np.diag(h))[:,None]),axis=0),6)
for key in ('panel_b','panel_c'):
 p=s[key]; u=p['u']; cols=u.T; a=np.c_[np.ones(len(u)),u]; q=np.column_stack([u[:,i]*u[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)])
 for k in p:
  if k=='u':continue
  v=p[k]; print(key,k, 'affine',report(a,v),'quad',report(np.c_[a,q],v)); print('affine coef',np.linalg.lstsq(a,v,rcond=None)[0].round(5)); print('quad coef',np.linalg.lstsq(np.c_[a,q],v,rcond=None)[0].round(5))
  for name,V in [('sin',np.sin),('tanh',np.tanh),('cos',np.cos)]:
   for f in (0.5,1,1.5,2,3):
    print(name,f,report(np.c_[a,q,V(f*u)],v))
