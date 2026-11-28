from campaign_probe import *
np.set_printoptions(precision=5,suppress=True,linewidth=150)
p=s['panel_a']; u,v=p['u'],p['v']; a=aff(u); k=np.argsort(np.linalg.norm(v-a@fit(a,v),axis=1))[:-3]; j=np.arange(4); h=(j+1)%4
for label in ('product','square'):
 q=u[:,h]; ds=np.stack((np.ones_like(u),u,q,u*q if label=='product' else q*q,u**3),axis=-1); print(label,np.array([fit(ds[k,i],v[k,i]) for i in j]))
p=s['panel_b']; u,v=p['u'],p['v']; print('B models')
for label,fn,init in [('gaussian',lambda u,p:u*(1-p[4]*np.exp(-p[:4]*u*u)),[1]*4+[1.1]),('gaussian+lin',lambda u,p:np.c_[u,u*np.exp(-p*u*u)],[1]*4),('tanh',lambda u,p:np.c_[u,np.tanh(p*u)],[2]*4),('rational',lambda u,p:np.c_[u,u/(1+p*u*u)],[1]*4),('sin',lambda u,p:np.c_[u,np.sin(p*u)],[2]*4),('sinonly',lambda u,p:np.sin(p*u),[2]*4)]:
 def r(q):
  a=aff(fn(u,q));return (a@fit(a,v)-v).ravel()
 z=least_squares(r,init,bounds=(.03,8),gtol=1e-13); print(label,np.std(r(z.x)),z.x)
 print(fit(aff(fn(u,z.x)),v).T)
print('time',times)
U,S,V=np.linalg.svd(tr['traj'],full_matrices=False)
for name,f in [('log',lambda p:np.log1p(p*times)),('exp',lambda p:-np.expm1(-p*times)),('power',lambda p:times**p),('sqrt',lambda p:np.sqrt(1+p*times)-1)]:
 for num in (1,2):
  def fun(p):
   a=np.column_stack([f(p)]+([np.ones_like(times)] if num==2 else []));return np.linalg.norm(V[0]-a@fit(a,V[0]))
  z=minimize_scalar(fun,bounds=(.001,15),method='bounded'); print(name,num,z.x,z.fun)
z,b,c,d=old._coords(s); n=52; Q=np.c_[aff(z[:n]),b[:n],c[:n],d[:n]]; T=tr['traj'][:n]; y=tr['y'][:n]; res=T-Q@fit(Q,T); print('state residual svd',np.linalg.svd(res,compute_uv=False)); print('time residual',np.linalg.svd(res,full_matrices=False)[2][:2]);
amp=old._shared_amplitude(times,T,Q); print('linear contrast',np.std(y-amp-Q@fit(Q,y-amp)), 'linear state',np.std(res));
F=np.c_[aff(z),b,(z[:,:,None]*b[:,None,:]).reshape(len(z),-1)]; A=fit(F[:n],np.c_[tr['y'][:n],T]); print('bilinear coeffs y / final state'); print(A[9:,[0,-1]].reshape(4,4,2)); print('train bilinear',np.std(F[:n]@A-np.c_[tr['y'][:n],T],axis=0));
