exec(open('/tmp/probe11.py').read().split("a=s['panel_a']")[0])
a=s['panel_a']; u,v=a['u'],a['v']; good=np.max(abs(u-v),axis=1)<3; u,v=u[good],v[good]; j=np.arange(4)
F={'sin':np.sin,'tanh':np.tanh,'oddRat':lambda a:a/(1+a*a),'soft':lambda a:a/(1+abs(a)),'cub':lambda a:a**3,'atan':np.arctan}
fits=[]
for name,fn in F.items():
 for own in ['cub','sin','tanh','oddRat','soft','atan']:
  def g(z,k):
   o=np.roll(z,2,axis=1); return np.stack([np.ones_like(z),z,F[own](k[0]*z),fn(k[1]*o)],axis=-1)
  def fit(k):
   f=g(u,k); w=np.array([ls(f[:,i],v[:,i]) for i in j]); return (np.einsum('njd,jd->nj',f,w)-v).ravel(),w
  opt=min([least_squares(lambda p:fit(p)[0],[.8,k],bounds=([.01,.01],[6,6]),ftol=1e-10) for k in [.5,1.2,2.15,3.]],key=lambda x:x.cost)
  fits.append((np.std(opt.fun),name,own,opt.x,fit(opt.x)[1]))
for z in sorted(fits,key=lambda a:a[0])[:12]:print(z)
_,name,own,k,w=min(fits,key=lambda a:a[0]);
def forward(z):return w[:,0]+w[:,1]*z+w[:,2]*F[own](k[0]*z)+w[:,3]*F[name](k[1]*np.roll(z,2,axis=1))
z=np.array([least_squares(lambda z:forward(z[None,:])[0]-x,(x-w[:,0])/w[:,1],gtol=1e-12).x for x in tr['x']]); print('inverse',np.max(abs(forward(z)-tr['x'])),z.min(0),z.max(0))
cs=s['panel_c']; u=cs['u']; ops=[]
for key,fn,k0 in [('v1',lambda x,k:x/(1+k*abs(x)),.83),('v2',lambda x,k:np.log1p(k*x*x),.36)]:
 v=cs[key]
 def fit(k):
  f=D(u,fn(u,k)); return (f@ls(f,v)-v).ravel()
 k=least_squares(fit,[k0],bounds=(.01,4)).x; w=ls(D(u,fn(u,k)),v); print(key,k,np.round(w,5)); ops.append(D(z,fn(z,k))@w)
p,r=ops; dot=np.sum(p*r,axis=1); off=np.column_stack([p[:,i]*r[:,j] for i in range(4) for j in range(4) if i!=j]); cross=(p[:,:,None]*r[:,None,:]).reshape(len(z),-1)
def tm(p):return -np.expm1(-t[:,None]*p)
rates=least_squares(lambda p:(tm(p)@ls(tm(p),tr['traj'][:52].T)-tr['traj'][:52].T).ravel(),[1.8,.5],bounds=([.8,.02],[8,.79])).x
basis=np.linalg.qr(tm(rates))[0]; scores=tr['traj']@basis; print('time',t,rates,np.std(scores@basis.T-tr['traj'])); np.savez('/tmp/sp11.npz',z=z,p=p,r=r,t=t)
for name,f in [('z',D(z)),('pr',D(p,r)),('common',D(p,r,dot)),('common+off',D(p,r,dot,off)),('common+xyz',D(p,r,dot,z,prod(z,3))),('full',D(p,r,z,prod(z,3),cross)),('common+zquad',D(p,r,dot,z,z*z,prod(z,2))),('z cubic',D(z,z*z,z**3,prod(z,2),prod(z,3)))]:
 yp=f@ls(f[:52],tr['y'][:52]); sp=(f@ls(f[:52],scores[:52]))@basis.T; print(name,np.round([np.mean(abs(yp[:52]-tr['y'][:52])),np.mean(abs(yp[52:]-tr['y'][52:])),np.mean(abs(sp[:52]-tr['traj'][:52])),np.mean(abs(sp[52:]-tr['traj'][52:]))],6))
