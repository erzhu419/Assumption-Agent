import numpy as np,pickle,itertools as it
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(s['panel_a'][k] for k in ('u','v')); m=np.linalg.norm(v-u,axis=1)<4; u,v=u[m],v[m]; d=u.shape[1]; np.set_printoptions(precision=4,suppress=True,linewidth=130)
def fit(F, detail=False):
 b=np.linalg.lstsq(F,v,rcond=None)[0]; r=v-F@b; h=(F*np.linalg.pinv(F).T).sum(1); cv=np.sqrt(np.mean((r/(1-h[:,None]))**2));
 if detail: print('rmse,loo',np.sqrt(np.mean(r*r)),cv,'coeffs',b,'sorted residuals',np.sort(np.linalg.norm(r,axis=1)))
 return cv
A=np.c_[np.ones(len(u)),u]; fit(A,True)
for deg in [2,3]:
 terms=[c for p in range(2,deg+1) for c in it.combinations_with_replacement(range(d),p)]; print('poly',deg,terms); fit(np.column_stack([A]+[np.prod(u[:,c],axis=1) for c in terms]),True)
for scale in [.22,.5,1.,1.25,1.5,2.,2.5,3.,3.5,4.,5.,6.]:
 for f in [np.sin,np.cos,np.tanh,np.arcsinh,np.abs]:
  print(scale,f.__name__,round(fit(np.c_[A,f(scale*u)]),5),end='; ')
 print()
r=v-A@np.linalg.lstsq(A,v,rcond=None)[0]; print('residual rows'); print(np.c_[u,r])
