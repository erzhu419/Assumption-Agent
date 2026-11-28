import sys,pickle
sys.path.insert(0,'/holdout_task')
from node_01 import calibrate,poly
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import orthogonal_mp, ElasticNet
from sklearn.model_selection import KFold
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z=calibrate(s['panel_a'],s['train']['x']); Y=np.c_[s['train']['y'],s['train']['traj']]; N=52
one=lambda x:np.ones(len(x))
P=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-9)[0]
F={'z':z}; kinds={'s':lambda x:np.c_[one(x),x,np.sin(x)],'sc':lambda x:np.c_[one(x),x,np.sin(x),np.cos(x)],'c':lambda x:np.c_[one(x),x,np.cos(x)],'sq':lambda x:np.c_[one(x),x*x,np.sin(x)],'t':lambda x:np.c_[one(x),x,np.tanh(x),np.tanh(x)**2], 'r':lambda x:np.c_[one(x),x,x/(1+x*x),x*x/(1+x*x)]}
for k,j in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[k]['u'],s[k][j]; out=[]
 for name,fn in kinds.items():
  def cost(g):
   a=fn(g*u); q=np.linalg.qr(a)[0]; return np.mean(((v-a@P(a,v))/np.maximum(.002,1-(q*q).sum(1))[:,None])**2)**.5
  m=min([minimize_scalar(cost,bounds=(a,b),method='bounded') for a,b in [(0.4,1.8),(1.8,3.3),(3.3,5)]],key=lambda x:x.fun); out.append((m.fun,name,m.x))
 print(k,j,sorted(out),flush=True)
 for _,name,g in sorted(out)[:2]:
  fn=kinds[name]; B=P(fn(g*u),v); print(name,np.round(B,3),flush=True)
 _,name,g=min(out); fn=kinds[name]; F[j]=fn(g*z)@P(fn(g*u),v)
np.savez('/tmp/grammar.npz',**F)
splits=[(np.arange(N),np.arange(N,len(z)))]+list(KFold(4,shuffle=True,random_state=23).split(z)); result=[]
def trial(name,A):
 for reg in (0.,.0001,.003,.06,.6):
  errs=[]
  for tr,va in splits:
   mu=A[tr].mean(0); std=A[tr].std(0).clip(.000004); D=(A-mu)/std; m=Y[tr].mean(0); V=Y[tr]-m
   if reg: pred=(D[va]@D[tr].T)@np.linalg.solve(D[tr]@D[tr].T+reg*np.eye(len(tr)),V)+m
   else: pred=D[va]@P(D[tr],V)+m
   E=abs(pred-Y[va]); errs.append([E[:,0].mean(),E[:,1:].mean()]);
  result.append((name,reg,np.round(errs[0],4),np.round(np.mean(errs[1:],0),4)))
for k in range(1,5):
 for names in combinations(F,k):
  W=np.column_stack([F[j] for j in names]); trial(str(names)+' quadratic',poly(W,2)); trial(str(names)+' additiveQ',np.column_stack([poly(F[j],2) for j in names])); trial(str(names)+' additiveC',np.column_stack([poly(F[j],3) for j in names]));
for f in (1.25,2,2.5,3,3.2,3.5,4):
 b=np.c_[z,np.sin(f*z),np.cos(f*z)]; trial('trigq '+str(f),poly(b,2)); trial('trig sep '+str(f),np.c_[poly(z,2),np.sin(f*z),np.cos(f*z),z**3,z**4]);
for i in range(2):
 print('best',i,*sorted(result,key=lambda v:v[2][i]+v[3][i])[:12],sep='\n')
