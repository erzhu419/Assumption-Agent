import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=5,suppress=True,linewidth=145)
ls=lambda a,b: np.linalg.lstsq(a,b,rcond=1e-10)[0]
one=lambda a: np.c_[np.ones(len(a)),a]
u,v=[s['panel_a'][k] for k in ('u','v')]; keep=np.linalg.norm(v,axis=1)<4
print('retained',sum(keep))
for degree in (1,2,3):
 terms=[t for k in range(1,degree+1) for t in itertools.combinations_with_replacement(range(4),k)]; A=one(np.array([u[:,t].prod(1) for t in terms]).T); c=ls(A[keep],v[keep]); r=A@c-v
 print('degree',degree,'rmse',np.sqrt(np.mean(r[keep]**2)),'loo rmse',np.sqrt(np.mean((r[keep]/(1-np.sum(A[keep]*np.linalg.pinv(A[keep]).T,axis=1)[:,None]))**2))); print(list(zip(['1']+[''.join(map(str,t)) for t in terms],c.round(5))))
for fn in [lambda u:u,lambda u:np.c_[u,u**2],lambda u:np.c_[u,u*np.roll(u,-1,axis=1)],lambda u:np.c_[u,u**3],lambda u:np.c_[u,u**2,u**3]]:
 A=one(fn(u)); c=ls(A[keep],v[keep]); print('reduced',A.shape[1],np.sqrt(np.mean((A[keep]@c-v[keep])**2)))
p=s['panel_c']; u=p['u']
for k,fun,q,b in [('v1 softsign',lambda u,q:one(u/(1+q[0]*abs(u))+q[1]*u),[.8,0],([.01,-2],[8,2])),('v2 curvature',lambda u,q:one(np.c_[u,u*u,u*u/(1+q[0]*u*u)]),[.3],([.01],[5])),('v1 cyclic',lambda u,q:one(u*np.roll(u,-1,axis=1)),[1],([.01],[8])),('v2 cosine',lambda u,q:one(np.cos(q[0]*u)),[2.4],([.01],[8])),('v1 arctan',lambda u,q:one(np.c_[u,u**3,np.arctan(q[0]*u)]),[1],([.01],[8]))]:
 y=p[k.split()[0]]; res=lambda q:(fun(u,q)@ls(fun(u,q),y)-y).ravel(); z=least_squares(res,q,bounds=b,gtol=1e-13,ftol=1e-13,xtol=1e-13).x;print(k,z,np.std(res(z)))
