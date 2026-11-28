import sys,pickle,numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
sys.path.insert(0,'/holdout_task/_frozen_nodes')
import node_00 as a,node_02 as b
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; n=len(t['y']); ix=np.arange(n)
splits=[np.arange(52,n)]+[j for seed in (102,701,208,330,12,718) for j in np.array_split(np.random.default_rng(seed).permutation(n),4)]
rows=[]
for j in splits:
 i=np.setdiff1d(ix,j); p=a.run_node({'__input__':dict(s,train={k:t[k][i] for k in ('x','y','traj')},test={'x':t['x'][j]})}); r=b.run_node({'node_00':p}); z=p['scores']; q=r['state_pred']@p['modes'].T; K=p['K']+.01*(z@z.T); Q=p['Kq']+.01*(q@z.T); y=t['y'][i]; yt=t['y'][j]; res={}
 def test(key,F,G,rr,adapt=False):
  ev,U=np.linalg.eigh(F); wy=U.T@y
  if adapt:
   def loss(logr):
    h=1/(ev+np.exp(logr)); e=(U@(wy*h))/((U*U)@h); return np.mean(e*e)
   rr=[np.exp(minimize_scalar(loss,bounds=(np.log(rr[0]),np.log(rr[1])),method='bounded').x)]
  GU=G@U
  for ridge in rr:
   yh=GU@(wy/(ev+ridge)); res[key,'adapt' if adapt else ridge]=np.mean(abs(yh-yt))
 test('base',K,Q,[.002,.004,.006,.008,.01,.015,.02,.03,.04])
 test('adaptive',K,Q,[.003,.03],True)
 for name,f,g in [('st',p['st'],p['qt']),('sy',p['sy'],p['qy']),('mix',np.c_[p['st'],p['sy']],np.c_[p['qt'],p['qy']])]:
  for w in [.03,.3,3.]:
   test((name,w),K+w*(f@f.T),Q+w*(g@f.T),[.004,.008,.015,.03])
  for fac in [.2,1.]:
   F=np.exp(-cdist(f,f,'sqeuclidean')/(fac*f.shape[1]));G=np.exp(-cdist(g,f,'sqeuclidean')/(fac*f.shape[1]));test((name,'rbf',fac),K+.1*F,Q+.1*G,[.004,.008,.015,.03])
 for power in [2,3,4]:
  norm=K.diagonal().mean(); test(('power',power),K+.3*K**power/norm**(power-1),Q+.3*Q**power/norm**(power-1),[.002,.004,.008,.015])
 rows.append(res)
for block in (13,len(rows)):
 r=rows[:block]; vals=sorted([(max(z[k] for z in r),np.mean([z[k] for z in r]),r[0][k],k) for k in r[0]])
 print('block',block,'best',*vals[:12],sep='\n'); print('base',*[v for v in vals if v[-1][0] in ('base','adaptive')],sep='\n')
