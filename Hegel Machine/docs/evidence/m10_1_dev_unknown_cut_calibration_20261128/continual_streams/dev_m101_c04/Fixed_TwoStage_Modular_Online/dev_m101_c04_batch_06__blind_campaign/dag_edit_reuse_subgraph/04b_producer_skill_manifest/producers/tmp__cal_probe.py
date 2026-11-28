import pickle,numpy as np,itertools
from scipy.linalg import qr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
u,v=(s['panel_a'][k] for k in ('u','v'))
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
a=np.c_[np.ones(len(u)),u]
# The gross errors are row-wise; select the consensus, then compare clean residuals.
m=np.ones(len(u),bool)
for _ in range(6):
 r=np.linalg.norm(a@ls(a[m],v[m])-v,axis=1); m=r<=np.sort(r)[44]
print('outlier rows',np.where(~m)[0], 'residual norms', np.round(np.sort(r)[-8:],4))
x=u[m]; y=v[m]; A=a[m];
q=list(itertools.combinations_with_replacement(range(4),2)); c=list(itertools.combinations_with_replacement(range(4),3));
Q=np.array([x[:,i]*x[:,j] for i,j in q]).T; C=np.array([x[:,i]*x[:,j]*x[:,k] for i,j,k in c]).T
for name,D in [('affine',A),('quadratic',np.c_[A,Q]),('cubic',np.c_[A,Q,C])]+[(f'{f.__name__}:{b}',np.c_[A,Q,f(b*x)]) for f in (np.tanh,np.sin,np.cos) for b in (.5,1.,1.5,2.,2.5,3.)]:
 h=np.linalg.qr(D)[0]; e=y-D@ls(D,y); loo=e/(1-np.sum(h*h,axis=1))[:,None];print(name,'rms',np.round(np.sqrt(np.mean(e**2,axis=0)),6),'loo',np.round(np.mean(abs(loo),axis=0),6))
 if name in ('quadratic','cubic'):
  print('coef',ls(D,y).round(3))
for k in ('panel_b','panel_c'):
 d=s[k]; x=d['u']; D=np.c_[np.ones(len(x)),x]
 for n in d:
  if n=='u':continue
  p=ls(D,d[n]);print(k,n,'coef',p.round(4),'rms',np.sqrt(np.mean((D@p-d[n])**2)))
