from probe_graph import *
from scipy.optimize import least_squares
p=S['panel_c']; u=p['u']; x=ops(S,T['x'])['x']; print('times',t,'telemetry',S['telemetry_vec'])
def A(z):return np.c_[np.ones(len(z)),z[:,0]*z[:,1],z[:,0]*z[:,3],z[:,1]*z[:,2],z[:,2]*z[:,3]]
c=A(x)@ls(A(u),p['v1'])
def B(z,w):return np.c_[poly(z,1),np.cos(z*w)]
def res(w):
 f=B(u,w);return (f@ls(f,p['v2'])-p['v2']).ravel()
w=least_squares(res,np.full(4,1.29),bounds=(.6,2)).x
d=B(x,w)@ls(B(u,w),p['v2']); m=c*d; k=c[:,0]*d[:,1]; one=np.ones(len(x));
L=np.c_[one,c,d,m]; M=np.c_[one,c,d,m.sum(1)]; G=np.c_[c,m]; H=np.c_[d,k]; HH=np.c_[G,H];
def grouped(a,b,shared,intercept):
 g=np.c_[c,m.sum(1)] if shared else G; h=H
 if intercept:g=np.c_[one,g]
 yy=T['traj'][a]; a0,b0=g[a],h[a]; start=ls(np.c_[a0,b0],yy); U=[]
 for s in [start[:a0.shape[1]],start[a0.shape[1]:]]:U.append(np.linalg.svd(s,full_matrices=False)[2][0])
 f,j=U
 for _ in range(50):
  z=np.c_[(a0[:,:,None]*f).transpose((0,2,1)).reshape(-1,a0.shape[1]),(b0[:,:,None]*j).transpose((0,2,1)).reshape(-1,b0.shape[1])]; v=ls(z,yy.ravel()); R=np.c_[a0@v[:a0.shape[1]],b0@v[a0.shape[1]:]]; f,j=ls(R,yy)
 return np.c_[g[b]@v[:a0.shape[1]],h[b]@v[a0.shape[1]:]]@np.array([f,j])
splits=[(np.arange(52),np.arange(52,72))]+[(p[:56],p[56:]) for p in [np.random.default_rng(i+23).permutation(72) for i in range(12)]]
for j in range(4):
 q=np.argsort(x[:,j]);splits += [(q[16:],q[:16]),(q[:-16],q[-16:])]
for name,F in [('full',L),('shared',M),('shared_zero',M[:,1:]),('full_zero',L[:,1:])]:
 e=[]
 for a,b in splits:e.append(np.mean(abs(F[b]@ls(F[a],T['y'][a])-T['y'][b])))
 print('y',name,e[0],max(e[1:13]),max(e[13:]),ls(F,T['y']))
for name in ['direct','rank2','grouped','grouped_shared','grouped_intercept','grouped_shared_intercept']:
 e=[]
 for a,b in splits:
  if name.startswith('grouped'):pred=grouped(a,b,'shared' in name,'intercept' in name)
  else:
   F=np.c_[one,HH]; pred=F[b]@ls(F[a],T['traj'][a]);
   if name=='rank2':
    v=np.linalg.svd(T['traj'][a],full_matrices=False)[2][:2];pred=(pred@v.T)@v
  err=abs(pred-T['traj'][b]); e.append([err.mean(),err.mean(0).max()])
 print('traj',name,e[0],np.max(e[1:13],0),np.max(e[13:],0))
