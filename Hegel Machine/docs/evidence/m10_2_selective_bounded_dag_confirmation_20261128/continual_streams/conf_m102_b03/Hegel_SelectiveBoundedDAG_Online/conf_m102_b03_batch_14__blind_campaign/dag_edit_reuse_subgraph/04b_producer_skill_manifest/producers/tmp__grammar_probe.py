import pickle,numpy as np,itertools as it
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
p=s['panel_a']; u,v=p['u'],p['v']; i,j=np.triu_indices(len(u),1)
for k in range(4):
 d=u[i,k]-u[j,k]; m=np.median((v[i,k]-v[j,k])/d); a=np.median(v[:,k]-m*u[:,k]); r=v[:,k]-m*u[:,k]-a; print('ALIGN',k,m,a,np.quantile(abs(r),[.25,.5,.75,.85,.9,1]))
u=s['panel_c']['u']; n=len(u); o=np.ones((n,1)); pp=np.array([u[:,i]*u[:,j] for i in range(4) for j in range(i,4)]).T
for k in [1,2]:
 v=s['panel_c']['v'+str(k)];
 def check(name,A):
  B=np.linalg.lstsq(A,v,rcond=None)[0]; inv=np.linalg.pinv(A); H=(A*inv.T).sum(1); R=A@B-v; print(k,name,round(np.mean(R**2)**.5,8),round(np.mean((R/np.maximum(1-H[:,None],1e-10))**2)**.5,8))
 for name,A in [('affine',np.c_[o,u]),('products',np.c_[o,pp]),('fullquad',np.c_[o,u,pp]),('archived products',np.c_[o,u[:,[0,0,1,2]]*u[:,[1,3,2,3]]]),('cube',np.c_[o,u,u**3]),('squares',np.c_[o,u,u**2]),('cos2.4',np.c_[o,u,np.cos(2.4*u)]),('cos1.27',np.c_[o,u,np.cos(1.27*u)]),('rational2',np.c_[o,u,u*u/(1+.165*u*u)])]:check(name,A)
 for nm,g in [('cos',np.cos),('sin',np.sin),('softsign',lambda w:w/(1+abs(w))),('rational2',lambda w:w*w/(1+w*w)),('rational1',lambda w:w/(1+w*w)),('tanh',np.tanh)]:
  h=min(np.arange(.2,10,.05),key=lambda h: np.linalg.norm((A:=np.c_[o,u,g(h*u)])@np.linalg.lstsq(A,v,rcond=None)[0]-v));check(f'{nm} grid {h:.2f}',np.c_[o,u,g(h*u)])
 B=np.linalg.lstsq(np.c_[o,u,pp],v,rcond=None)[0];print('quad-norms',np.round(np.linalg.norm(B,axis=1),3));print('SVD',np.linalg.svd(v,compute_uv=False))
