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
from itertools import combinations,product
z=np.load('/tmp/cubic_probe.npz')['z']; tr=np.arange(52); te=np.arange(52,72)
for k in ['v1','v2']:
 c=s['panel_c']; a=c['u']; b=c[k]; print(k, 'cubic axis', np.round(np.linalg.lstsq(np.c_[poly(a,1),a*a,a**3],b,rcond=None)[0],3));print('err',np.std(b-np.c_[poly(a,1),a*a,a**3]@np.linalg.lstsq(np.c_[poly(a,1),a*a,a**3],b,rcond=None)[0]))
print('telemetry',s['telemetry_vec']); print('times',s['eval_times']);
D=np.vstack([np.eye(4)]+[np.eye(4)[i]+sg*np.eye(4)[j] for i,j in combinations(range(4),2) for sg in [-1,1]]+[np.array([1,*sg]) for sg in product([-1,1],repeat=3)]); print('D', D)
for sq in [0,.09,.22,.5,1.25]:
 h=z/np.sqrt(1+sq*z*z);p=poly(h,2)
 for head,t in [('y',Y),('a',targets[:,1]),('b',targets[:,2])]:
  for f in [1.25,1.75,2.15,2.5,3.15]:
   A=np.c_[p,np.sin(f*h),np.cos(f*h)]; fit=np.linalg.lstsq(A[tr],t[tr],rcond=None)[0];r=t-A@fit
   cand=[];w=h@D.T
   for ff in [.5,1,1.25,1.75,2.15,2.5,3,4]:
    for nm,F in [('sin',np.sin(ff*w)),('cos',np.cos(ff*w)),('tanh',np.tanh(ff*w)),('rat',w/(1+ff*w*w)),('rat2',1/(1+ff*w*w))]:
     F-=A@np.linalg.lstsq(A[tr],F[tr],rcond=None)[0]; score=np.abs(r[tr]@F[tr])/np.maximum(np.linalg.norm(F[tr],axis=0),1e-8)
     j=score.argmax(); pred=F[:,j]*(r[tr]@F[tr,j])/max(np.sum(F[tr,j]**2),1e-10); cand.append((score[j],np.mean(abs(r[te]-pred[te])),nm,ff,j))
   if head=='y' or (sq==.09 and f==1.75):print(head,sq,f,'base',np.round([np.std(r[tr]),np.mean(abs(r[te]))],4),'terms', [tuple(np.round(q[:2],4))+q[2:] for q in sorted(cand,reverse=True)[:4]])
