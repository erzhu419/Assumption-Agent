import numpy as np,pickle
from itertools import combinations_with_replacement
from scipy.linalg import lstsq
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; X=tr['x']; Y=np.c_[tr['y'],tr['traj']]
np.set_printoptions(precision=4,suppress=True,linewidth=140)
def aff(x):return np.c_[np.ones(len(x)),x]
def fit(a,b):return lstsq(a,b)[0]
def pol(x,d):return np.column_stack([np.prod(x[:,i],axis=1) for n in range(d+1) for i in combinations_with_replacement(range(x.shape[1]),n)])
a=s['panel_a']; u,v=a['u'],a['v']; b=fit(aff(u),v); ix=np.argsort(np.linalg.norm(v-aff(u)@b,axis=1))[:-3]; b=fit(aff(u[ix]),v[ix]); print('times',s['eval_times'],'tel',s['telemetry_vec'],'A',b,'clean rms',np.std(v[ix]-aff(u[ix])@b,axis=0)); z=(X-b[0])@np.linalg.inv(b[1:]); print('z range',z.min(0),z.max(0)); print('target singular values',np.linalg.svd(Y[:,1:],compute_uv=False))
for k in ('panel_a','panel_b','panel_c'):
 p=s[k]; u=p['u']; ids=ix if k=='panel_a' else np.arange(len(u)); u=u[ids];
 for name,v in list(p.items())[1:]:
  v=v[ids]; print(k,name)
  for d in (1,2,3):
   a=pol(u,d); pred=a@fit(a,v); h=np.sum(np.linalg.svd(a,full_matrices=False)[0][:,:min(a.shape)]**2,axis=1); print(d,np.std(v-pred,axis=0),np.mean(np.abs((v-pred)/np.maximum(1-h[:,None],1e-12)),axis=0))
  r=v-aff(u)@fit(aff(u),v); q=pol(u,2)[:,5:]; q-=aff(u)@fit(aff(u),q); print('quadratic',fit(q,r))
def check(a):
 out=[]
 for i,j in [(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72)),(np.arange(20,72),np.arange(20))]:
  r=np.abs(a[j]@fit(a[i],Y[i])-Y[j]); out.append(np.r_[r[:,0].mean(),r[:,1:].mean(),r[:,-1].mean()]);
 return np.array(out)
for d in range(1,5): print('poly',d,check(pol(z,d)))
for scale in (.3,.6,1,1.6):
 for reg in (.0001,.01,.1):
  K=np.exp(-cdist(z,z,'sqeuclidean')/(2*scale**2)); out=[]
  for i,j in [(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72)),(np.arange(20,72),np.arange(20))]:
   r=np.abs(K[np.ix_(j,i)]@np.linalg.solve(K[np.ix_(i,i)]+reg*np.eye(len(i)),Y[i])-Y[j]); out.append(np.r_[r[:,0].mean(),r[:,1:].mean()]);
  print('rbf',scale,reg,np.array(out).ravel())
