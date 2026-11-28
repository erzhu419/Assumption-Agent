import numpy as np,itertools as it
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
D=dict(np.load('/tmp/expanded.npz')); Y=D['Y']; Q=D['Q']; z=D['z']; C=D['C']; H=D['D']; n=52
f=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-8)[0]; aff=lambda x:np.c_[np.ones(len(x)),x]; cyc=lambda x:x*np.roll(x,-1,axis=1)
features={}; groups={}
def add(k,a,solo=True):
 a=np.asarray(a).reshape(len(z),-1); groups[k]=a
 if solo:
  for i,v in enumerate(a.T):features[k+':'+str(i)]=v
for key in ['z','raw','B','Bi','BT']:
 x=D[key];L=np.stack([x[:,i]+s*x[:,j] for i,j in it.combinations(range(4),2) for s in [1,-1]],axis=1); c=aff(cyc(x))@C; d=aff(np.cos(2.4*x))@H; F={'x':x,'c':c,'d':d,'sumdiff':L};
 for j,a in F.items():
  if j!='sumdiff':add(key+j,a)
  for w in [1.25,2.4,3.0,4.8]:
   add(f'{key}{j}cos{w}',np.cos(w*a));add(f'{key}{j}sin{w}',np.sin(w*a))
  for name,fn in [('cube',lambda q:q**3),('sat',lambda q:q/(1+.22*q*q)),('tanh',np.tanh),('sq',np.square)]:add(key+j+name,fn(a))
 for j,h in [('x','c'),('x','d'),('c','d'),('c','c'),('d','d')]:
  a,b=F[j],F[h]; add(key+j+h+'dot',np.sum(a*b,axis=1)); add(key+j+h+'prod',(a[:,:,None]*b[:,None,:]).reshape(len(z),-1));add(key+j+h+'rational',(a[:,:,None]/(1+.22*b[:,None,:]**2)).reshape(len(z),-1))
 for degree in [2,3,4]:
  add(key+'poly'+str(degree),np.column_stack([np.prod(x[:,p],axis=1) for p in it.combinations_with_replacement(range(4),degree)]))
 for w in [1.25,2.4,4.8]:
  p=cyc(x); add(key+'cyccos'+str(w),np.cos(w*p));add(key+'cycsin'+str(w),np.sin(w*p))
 # Apparatus normalization, both scalar and channel-wise.
 for akey,a in [('c',c),('d',d)]:
  for bkey,b in [('x',x),('c',c),('d',d),('cd',c*d),('dot',np.sum(c*d,axis=1,keepdims=True))]:
   for scale,val in [('diag',b*b),('norm',(b*b).sum(1,keepdims=True))]:add(key+akey+bkey+scale,a/(1+.22*val))
K=list(groups); M=list(features); F=np.column_stack(list(features.values())); Ft=F[:n]; mean=Ft.mean(0); sd=Ft.std(0); F=(F-mean)/np.maximum(sd,1e-10)
for j,y in enumerate(Y.T):
 print('TARGET',j,flush=True); order=np.argsort(abs((y[:n]-y[:n].mean())@F[:n]))[-15:][::-1];print('top correlated',[(M[i],round(np.corrcoef(y[:n],F[:n,i])[0,1],3)) for i in order]);R=[]
 for k,a in groups.items():
  X=aff(a); p=X@f(X[:n],y[:n]);R.append((np.mean(abs(p[:n]-y[:n])),np.mean(abs(p[n:]-y[n:])),k))
 print('groups',sorted(R)[:12],sorted(R,key=lambda p:p[1])[:8]);
 for initial in [[],[M.index('zcddot:0')],[M.index('zd:0'),M.index('zd:1'),M.index('zd:2'),M.index('zd:3')]]:
  chosen=initial[:]
  for i in range(16):
   X=aff(F[:,chosen]); U=np.linalg.qr(X[:n])[0]; r=y[:n]-U@(U.T@y[:n]); G=F[:n]-U@(U.T@F[:n]); norms=np.linalg.norm(G,axis=0); scores=np.where(norms>1e-3,abs(r@G)/np.maximum(norms,1e-8),0.); scores[chosen]=-1; k=np.argmax(scores);chosen.append(k)
   X=aff(F[:,chosen]);p=X@f(X[:n],y[:n]);
   if i in [2,5,9,15]:print('omp',i,round(np.mean(abs(p[:n]-y[:n])),4),round(np.mean(abs(p[n:]-y[n:])),4),[M[k] for k in chosen]);
np.savez('/tmp/dictionary.npz',F=F,Y=Y,Q=Q,names=M)
