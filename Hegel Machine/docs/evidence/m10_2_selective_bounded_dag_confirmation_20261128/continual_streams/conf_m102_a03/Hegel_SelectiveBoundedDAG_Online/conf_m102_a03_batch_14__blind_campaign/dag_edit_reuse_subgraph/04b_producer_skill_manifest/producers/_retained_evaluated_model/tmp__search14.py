exec(open('/tmp/model14.py').read().split("print('ranges'")[0])
import importlib.util
spec=importlib.util.spec_from_file_location('head','/holdout_task/_frozen_nodes/node_01.py');head=importlib.util.module_from_spec(spec);spec.loader.exec_module(head)
y=s['train']['y']; t=s['train']['traj']; q=np.linalg.svd(t[:52],full_matrices=False)[2][:2]; v=t@q.T
from scipy.optimize import least_squares
from scipy.linalg import qr
from itertools import combinations_with_replacement as cwr

def poly(z):
 i,j=np.triu_indices(z.shape[1]);return z[:,i]*z[:,j]

def report(name,A,tar):
 U,sv,V=np.linalg.svd(A[:52],full_matrices=False); X=A@V.T/np.maximum(sv,1e-9); best=None
 for l in np.logspace(-7,0,26):
  h=sv**2/(sv**2+l); w=(U.T@tar[:52])*h.reshape((-1,)+(1,)*(tar.ndim-1)); err=(U@w-tar[:52])/np.maximum(1-(U*U)@h,1e-5).reshape((-1,)+(1,)*(tar.ndim-1)); score=np.mean(abs(err)); pred=X[52:]@w
  if best is None or score<best[0]:best=(score,pred,l)
 e=best[1]-tar[52:]; return (round(best[0],4),round(np.mean(abs(e if tar.ndim==1 else e@q)),4),best[2])

maps={'x':x,'b':b,'c':c,'d':d}
base=np.c_[np.ones(n),x,b,c,d]; pd=np.c_[np.ones(n),x,b,c,d,poly(x),poly(b),poly(c),poly(d)];
for name,X in list(maps.items())+[('base',base),('pd',pd)]:
 for deg in (1,2,3):
  if deg>2 and X.shape[1]>4:continue
  A=np.c_[np.ones(n),X,*[np.column_stack([np.prod(X[:,jj],axis=1) for jj in cwr(range(X.shape[1]),i)]) for i in range(2,deg+1)]]
  print(name,deg,report(name,A,y),report(name,A,v),flush=True)
for combo in [np.c_[x,b],np.c_[x,c],np.c_[x,d],np.c_[c,d],np.c_[x,c,d],np.c_[x,b,c,d]]:
 for f in (.25,.5,1,2,4,8):
  dist=np.sum((combo[:,None]-combo[None,:])**2,axis=2)/combo.shape[1]; K=np.exp(-f*dist); U,sv,_=np.linalg.svd(K); A=np.c_[pd,U*sv**.5];print('kernel',combo.shape[1],f,report('',A,y),report('',A,v))
# Screen structured additive and cross-channel products; refit solely on the prefix.
def dictionary(x,b,c,d):
 n=len(x); mats=[]; names=[]
 def add(a,name):
  if a.ndim==1:a=a[:,None]
  mats.append(a); names.extend([f'{name}:{k}' for k in range(a.shape[1])])
 add(np.ones(n),'bias'); add(np.c_[x,b,c,d],'linear');
 for name,z in zip('xbcd',[x,b,c,d]):
  add(poly(z),name+'quad'); add(z**3,name+'cube');
  for f in np.arange(.5,6.51,.25):
   for kind,fn in [('sin',np.sin),('cos',np.cos),('atan',np.arctan),('tanh',np.tanh)]:add(fn(f*z),f'{name}{kind}{f}')
  pairs=np.array(list(__import__('itertools').combinations(range(4),2)))
  for sign in (-1,1):
   u=z[:,pairs[:,0]]+sign*z[:,pairs[:,1]]
   for f in (.5,1,1.25,1.5,2,2.15,2.5,3,4):
    add(np.sin(f*u),f'{name}pair{sign}sin{f}');add(np.cos(f*u),f'{name}pair{sign}cos{f}');add(np.tanh(f*u),f'{name}pair{sign}tanh{f}')
 for i,j in __import__('itertools').combinations(range(4),2):add((list(maps.values())[i][:,:,None]*list(maps.values())[j][:,None,:]).reshape(n,-1),f'cross{i}{j}')
 return np.column_stack(mats),np.array(names)
D,names=dictionary(x,b,c,d)
for nam,tar in [('y',y),('t',v[:,0])]:
 for key,core in [('x',np.c_[np.ones(n),x]),('b',np.c_[np.ones(n),b,poly(b)]),('c',np.c_[np.ones(n),c]),('cd',np.c_[np.ones(n),c,d]),('xc',np.c_[np.ones(n),x,c]),('xd',np.c_[np.ones(n),x,d])]:
  def sparse(A,T,ntrain,maxk=20):
   sel=[]; out=[]
   for k in range(maxk+1):
    M=np.c_[A,D[:,sel]]; U,sv,_=np.linalg.svd(M[:ntrain],full_matrices=False); U=U[:,sv>1e-7]; r=T[:ntrain]-U@(U.T@T[:ntrain]); loo=np.mean(abs(r/np.maximum(1-np.sum(U*U,axis=1),1e-7))); pred=M[ntrain:]@np.linalg.lstsq(M[:ntrain],T[:ntrain],rcond=1e-7)[0];out.append((loo,np.mean(abs(pred-T[ntrain:])),sel.copy())); norms=np.sqrt(np.maximum(0,np.sum(D[:ntrain]**2,axis=0)-np.sum((U.T@D[:ntrain])**2,axis=0))); score=abs(r@D[:ntrain])/np.maximum(norms,1e-6); score[norms<.005]=0; sel.append(int(np.argmax(score)))
   return min(out,key=lambda l:l[0]),min(out,key=lambda l:l[1]);
  aa,bb=sparse(core,tar,52);print('sparse',nam,key,(aa[:2],names[aa[2]].tolist()),(bb[:2],names[bb[2]].tolist()),flush=True)
