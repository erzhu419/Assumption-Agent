from probe import *
from itertools import combinations
from sklearn.linear_model import Lasso
z,b,_,w=coords(T['x']); p=S['panel_c']; edges=[(0,1),(0,3),(1,2),(2,3)]
def edge(x):return np.column_stack([x[:,i]*x[:,j] for i,j in edges])
c=aff(edge(z))@fit(aff(edge(p['u'])),p['v1']); D,A,r=modes(T['traj'][:52]); A=fit(D,T['traj'].T).T; Y=np.c_[T['y'],A]; n=52
base=aff(np.c_[power(z,2),c*w]); be=fit(base[:n],Y[:n]); print('base coeffs',np.round(be,4)); R=Y-base@be; print('base residual train,test',np.mean(abs(R[:n]),0),np.mean(abs(R[n:]),0))
F={};
def put(k,f):F[k]=np.asarray(f).reshape(len(z),-1)
for k,q in [('z',z),('b',b),('c',c),('w',w),('c*w',c*w),('z+w',z+w),('z-w',z-w),('z+c',z+c),('z-c',z-c),('z*b',z*b),('edge',edge(z)),('pair+',np.column_stack([z[:,i]+z[:,j] for i,j in combinations(range(4),2)])),('pair-',np.column_stack([z[:,i]-z[:,j] for i,j in combinations(range(4),2)]))]:
 put(k,q);put(k+'sq',q*q);put(k+'cube',q**3)
 for a in (.3,.6,1,1.5,2):
  for f,fn in [('sin',np.sin),('tanh',np.tanh),('cos',np.cos),('rat',lambda x:x/(1+x*x)),('inv',lambda x:1/(1+x*x))]:
   v=fn(a*q);put(f'{k}:{f}:{a}',v)
   for nm,l in [('z',z),('w',w),('c',c)]:
    if q.shape[1]==4:put(f'{k}:{f}:{a}*{nm}',v*l)
for k,q in [('b',b),('c',c),('w',w),('z',z)]:
 for j,a in [('b',b),('c',c),('w',w),('z',z)]:
  if j!=k:put(f'{k}x{j}',(q[:,:,None]*a[:,None,:]))
put('cubic',power(z,3)[:,14:]);put('quartic',power(z,4)[:,34:]);
def assess(B,F,y):
 results=[]
 for k,f in F.items():
  a=np.c_[B,f]; u,s,v=np.linalg.svd(a[:n],full_matrices=False); rank=s>s[0]*1e-7; u=u[:,rank]; bh=(v[rank].T/s[rank])@(u.T@y[:n]); e=y-a@bh; h=np.sum(u*u,1); loo=np.mean(abs(e[:n]/(1-h))); results.append((loo,k,np.mean(abs(e[n:])),np.mean(abs(e[:n]))))
 return sorted(results)
for j in range(3):
 a=assess(base,F,Y[:,j]); print('best',j,a[:12]);
 if j==1:
  for _,k,_,_ in a[:3]:print('next',k,assess(np.c_[base,F[k]],{s:v for s,v in F.items() if s!=k},Y[:,j])[:4])
# Compare the slow-mode baseline on the actual trajectory, not only on its noisy modal coefficient.
for k in ['base','cubic']+[r[1] for r in assess(base,F,Y[:,1])[:5]]:
 x=base if k=='base' else np.c_[base,F[k]]; fast=base@be[:,2]; slow=x@fit(x[:n],Y[:n,1]); pred=np.c_[slow,fast]@D.T; print('trajectory',k,np.mean(abs(pred[n:]-T['traj'][n:])),np.round(np.mean(abs(pred[n:]-T['traj'][n:]),0),5))
