import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];x=s['train']['x'];Y=s['train']['y'];Z=s['train']['traj'];T=np.linalg.svd(Z,full_matrices=False)[2][:2]; targets=np.c_[Y,Z@T.T]
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for k in range(1,d+1) for j in combinations_with_replacement(range(x.shape[1]),k)])
def panel(u,v,fun):
 f=fun(u);keep=np.ones(len(u),bool)
 for j in range(5):
  b=np.linalg.lstsq(f[keep],v[keep],rcond=None)[0];e=np.linalg.norm(f@b-v,axis=1);keep=e<3
 return b,e
from itertools import product, combinations
z=np.load('/tmp/cubic_probe.npz')['z']; tr=np.arange(52); te=np.arange(52,72)
D=np.array([d for d in product([-1,0,1],repeat=4) if any(d) and next(x for x in d if x)==1]);
for squash in [0,.09,.22]:
 h=z/np.sqrt(1+squash*z*z); w=h@D.T
 F=[]; keys=[]
 def add(v,k):
  v=np.asarray(v).reshape(len(h),-1);F.append(v);keys.extend([(k,j) for j in range(v.shape[1])]);
 for d in [2,3]:
  add(poly(h,d),('poly',d))
 for f in [.5,1,1.25,1.75,2.15,2.5,3,4]:
  for nm,v in [('sin',np.sin(f*w)),('cos',np.cos(f*w)),('rat',w/(1+f*w*w)),('rat2',1/(1+f*w*w)),('tanh',np.tanh(f*w))]:add(v,(nm,f))
 for f in [.22,.5,1,2.15,3]:
  for j in range(4): add(h[:,j,None]/(1+f*w*w),('ratio',j,f))
  for j,k in combinations(range(4),2):add((h[:,j]*h[:,k])[:,None]/(1+f*w*w),('ratio2',j,k,f))
 F=np.column_stack(F)
 for base in [poly(h,1),poly(h,2),np.c_[poly(h,2),np.sin(2.15*h),np.cos(2.15*h)],np.c_[poly(h,2),np.sin(2.15*h),np.cos(1.25*h)]]:
  active=[]; A=base
  for step in range(9):
   q=np.linalg.svd(A[tr],full_matrices=False)[0];R=F[tr]-q@(q.T@F[tr]); norm=np.linalg.norm(R,axis=0); beta=np.linalg.lstsq(A[tr],Y[tr],rcond=None)[0]; e=Y-A@beta
   if step<5 or step==8: print(squash,base.shape[1],step,round(np.std(e[tr]),5),round(np.mean(abs(e[te])),5),[(keys[i],tuple(D[keys[i][1]]) if keys[i][0][0]!='poly' else None) for i in active]);
   j=np.argmax(np.where(norm>1e-7,abs(e[tr]@R)/np.maximum(norm,1e-9),0)); active.append(j); A=np.c_[A,F[:,j]]
c=s['panel_c'];a=c['u'];
for key in ['v1','v2']:
 res=[]
 for f in np.arange(.1,3,.05):
  for nm,p in [('sin',np.sin(f*a)),('tanh',np.tanh(f*a)),('rat',a/np.sqrt(1+f*a*a)),('sincos',np.c_[np.sin(f*a),np.cos(f*a)])]:
   A=np.c_[np.ones(len(a)),p];P=np.linalg.pinv(A); e=c[key]-A@(P@c[key]);res.append((np.mean((e/(1-np.sum(A*P.T,axis=1))[:,None])**2)**.5,nm,round(f,3)))
 print('C',key,sorted(res)[:8]);
