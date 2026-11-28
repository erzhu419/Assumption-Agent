import sys,pickle,numpy as np
from itertools import combinations
from scipy.spatial.distance import cdist
from sklearn.model_selection import ShuffleSplit
from scipy.linalg import cho_factor,cho_solve
sys.path.insert(0,'/holdout_task'); from batch_solver import calibrate,temporal
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; o,g=calibrate(s); z=(tr['x']-o)/g; y=tr['y']; Y=tr['traj']; t=s['eval_times']; n=len(z)
pairs=np.array(list(combinations(range(4),2))); c=z[:,pairs[:,0]]*z[:,pairs[:,1]]; C=c[:,[0,2,3,5]]; l=z@z.T; L=1+l; Q=L+c@c.T+(z*z)@(z*z).T; ck=C@C.T; d=cdist(z,z); H=np.sqrt(5)*d/2; M=(1+H+H*H/3)*np.exp(-H); I=np.eye(n)
S=np.column_stack([z*z/np.sqrt(1+.39*z*z),np.tanh(z),z/(1+.22*z*z)]); D=S@S.T
p=s['panel_c']; u=p['u']; F=np.c_[np.ones(len(u)),u[:,pairs[[0,2,3,5],0]]*u[:,pairs[[0,2,3,5],1]]]; print('C coefficients',np.linalg.lstsq(F,p['v1'],rcond=None)[0]);
splits=[(np.arange(52),np.arange(52,72)),*ShuffleSplit(24,train_size=52,random_state=11).split(z)]
R={}; predictions={}; specs={}
def add(name,K):
 vals=[]; ps=[]
 for a,b in splits:
  T=np.c_[y,Y]; C0=cho_factor(K[np.ix_(a,a)]); p=K[np.ix_(b,a)]@cho_solve(C0,T[a]); e=p-T[b]; vals.append([np.mean(abs(e[:,0])),np.mean(abs(e[:,1:])),np.mean(abs(e[:,-1]))]); ps.append(p)
 R[name]=np.array(vals); predictions[name]=ps
for alpha in [.04,.15,.45]:
 for ridge in [.002,.006,.018]:
  for mi in [0.,.25,1.]:
   for sep in [0.,.4]:
    name=(alpha,ridge,mi,sep); add(name,Q+alpha*ck*(L**2+mi*4*M)+sep*D+ridge*I)
add('cycle',Q+ck*M+.002*I); add('poly',Q+.03*l**3+.003*l**4+.002*I)
for j in range(3):
 print('metric',j)
 for name,r in sorted(R.items(),key=lambda kv:.65*np.mean(kv[1][:,j])+.35*np.max(kv[1][:,j]))[:8]:print(name,np.round([r[0,j],r[:,j].mean(),r[:,j].max()],5))
for name2 in ['cycle','poly']:
 for wt in [.15,.3,.5]:
  r=[]
  for (a,b),p1,p2 in zip(splits,predictions[.15,.002,0.,0.],predictions[name2]):
   e=(1-wt)*p1+wt*p2-np.c_[y,Y][b];r.append([np.mean(abs(e[:,0])),np.mean(abs(e[:,1:])),np.mean(abs(e[:,-1]))]);
  print('blend',name2,wt,np.round([r[0],np.mean(r,0),np.max(r,0)],5))
