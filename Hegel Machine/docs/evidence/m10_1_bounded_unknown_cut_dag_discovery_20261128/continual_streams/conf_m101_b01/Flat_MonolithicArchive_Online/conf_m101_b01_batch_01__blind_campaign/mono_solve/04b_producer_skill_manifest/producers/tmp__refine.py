from grammar import *
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from sklearn.model_selection import KFold
from importlib.util import spec_from_file_location,module_from_spec
spec=spec_from_file_location('baseline','/holdout_task/batch_solver.py'); old=module_from_spec(spec); spec.loader.exec_module(old)
X=old._calibrate(S['panel_a'],np.r_[T['x'],S['test']['x']]); n=len(Y)
u=S['panel_b']['u']; v=S['panel_b']['v']; uc=S['panel_c']['u']; vc=S['panel_c']['v1']; vd=S['panel_c']['v2']
def report(d,v):
 inv=np.linalg.pinv(d); e=v-d@(inv@v); h=np.sum(d*inv.T,axis=1); return np.round([np.std(e),np.std(e/(1-h[:,None]))],6)
for extra in ['none','sq','cross','biasnonlin','odd5','radial','squaresum']:
 def d(x,k):
  d=np.c_[aff(x),x*np.exp(-(k*x)**2)]
  a={'none':[], 'sq':[x*x], 'cross':[poly(x,2)[:,5:]],'biasnonlin':[np.exp(-(k*x)**2)],'odd5':[x**5], 'radial':[x*np.sum(x*x,axis=1,keepdims=True)],'squaresum':[np.sum(x*x,axis=1)]}[extra]
  return np.column_stack([d,*a])
 k=minimize_scalar(lambda k:report(d(u,k),v)[1],bounds=(.7,1.4),method='bounded').x; print('B',extra,k,report(d(u,k),v))
 if extra=='none':
  w=fit(d(u,k),v); e=v-d(u,k)@w; print('coef',np.round(w,4),'B residual SVD',np.linalg.svd(e,compute_uv=False),'quadratic residual projection',np.round(fit(poly(u,2),e)[5:],4))
base=old._coordinates(S,np.r_[T['x'],S['test']['x']]); Z={}
bs=panel_fit(u,v); cs=panel_fit(uc,vc)
for bn in ['exp','erf','tanh','cuberat','sin']:
 b=next(r for r in bs if r[1]==bn and r[2]=='mix'); z=b[-1](X)
 for cn in ['integral','atan']:
  c=next(r for r in cs if r[1]==cn and r[2]=='mix'); q=c[-1](z)
  ka=minimize_scalar(lambda k:np.mean((aff(uc+k*uc**2)@fit(aff(uc+k*uc**2),vd)-vd)**2),bounds=(.2,.7),method='bounded').x
  d=aff(z+ka*z**2)@fit(aff(uc+ka*uc**2),vd)
  Z[(bn,cn)]=(q,d,z)
print('C2 shared quadratic',ka,report(aff(uc+ka*uc**2),vd))
near=np.partition(cdist(X[:n],X[:n]),5,axis=1)[:,5]; rad=np.linalg.norm(X[:n],axis=1)
sp=[(np.arange(52),np.arange(52,n)), *KFold(4,shuffle=True,random_state=91).split(Y), *[(np.argsort(xx)[:-15],np.argsort(xx)[-15:]) for xx in (near,rad,*X[:n].T,*(-X[:n]).T)]]
target=np.c_[Y,Q]; ans=[]
for (bn,cn),(q,d,z) in Z.items():
 for mode in ['quad/aff','quad/diag','quad/dual','quad/c2rat','quad/sat','ridge','rbf','rbfdual','cubic/diag']:
  pair={'quad/aff':(poly(q,2),aff(q)),'quad/diag':(poly(q,2),np.c_[aff(q),q*q]),'quad/dual':(poly(q,2),np.c_[aff(q),d]),'quad/c2rat':(poly(q,2),np.c_[aff(q),np.tanh(d)]),'quad/sat':(poly(q,2),np.c_[aff(q),np.tanh(q),q*q]),'ridge':(poly(q,2),poly(q,2)),'cubic/diag':(poly(q,3),np.c_[aff(q),q*q])}.get(mode)
  f=q if mode!='rbfdual' else np.c_[q,d]; K=np.exp(-cdist(f,f,'sqeuclidean')/(2*f.shape[1]*2.4**2)); es=[]
  for i,j in sp:
   if pair:
    sy,st=pair
    def pred(a,b,l): return a[j]@solve(a[i].T@a[i]+l*np.eye(a.shape[1]),a[i].T@b[i],assume_a='pos')
    py=pred(sy,Y,.035 if mode=='cubic/diag' else .0001); pt=pred(st,Q,.025 if mode=='ridge' else .0001)
   else:
    p=target[i].mean(0)+K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+.0001*np.eye(len(i)),target[i]-target[i].mean(0),assume_a='pos'); py=p[:,0]; pt=p[:,1:]
   es.append([np.mean(abs(py-Y[j])),np.mean(abs(pt-Q[j]))])
  e=np.array(es); ans.append((bn,cn,mode,*[np.round(k,5) for k in (e[0],e[1:5].max(0),e[5:].mean(0),e[5:].max(0))]))
for row in sorted(ans,key=lambda x:np.mean(x[3])+.5*np.mean(x[4])+np.mean(x[5])+.3*np.mean(x[6]))[:20]:print(row)
for row in sorted(ans,key=lambda x:x[5][1]+.3*x[6][1])[:8]:print('traj',row)
