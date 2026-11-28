import numpy as np,runpy, pickle
from itertools import combinations_with_replacement
from scipy.optimize import least_squares,root
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; u,v=(s['panel_a'][k] for k in ['u','v']); good=np.linalg.norm(v-u,axis=1)<3; u,v=u[good],v[good]; ls=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]
def f(q,p):
 return [np.c_[np.ones(len(q)),q[:,j],np.tanh(p[0]*q[:,j]),np.sin(p[1]*q[:,(j+2)%4])] for j in range(4)]
def resid(p):
 return np.column_stack([d@ls(d,v[:,j])-v[:,j] for j,d in enumerate(f(u,p))]).ravel()
p=least_squares(resid,[1.8,2.4]).x; w=[ls(d,v[:,j]) for j,d in enumerate(f(u,p))]; print('cal',p,abs(resid(p)).mean(),w)
x=np.r_[tr['x'],s['test']['x']]; q=np.array([root(lambda row:np.array([(d@w[j])[0] for j,d in enumerate(f(row[None],p))])-r,r).x for r in x]); n0=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py'); upstream=n0['run_node']({'__input__':s}); b=n0['_b_features'](q,upstream['p'])@upstream['w']; c=b*np.roll(q,-1,axis=1); y=tr['y']; z=tr['traj']; t=s['telemetry_vec']; o=np.ones(len(q));
np.savez('/tmp/read11.npz',q=q,b=b,c=c,y=y,z=z,t=t)
folds=[np.arange(52,72)]+list(np.array_split(np.random.default_rng(42).permutation(72),5));
def score(d,y):
 rr=[]
 for te in folds:
  it=np.setdiff1d(np.arange(len(y)),te); rr.append(abs(d[te]@ls(d[it],y[it])-y[te]).mean())
 return (max(rr),rr[0],np.mean(rr),abs(d[:len(y)]@ls(d[:len(y)],y)-y).mean())
print('state',score(np.c_[o,b,c.sum(1)],z)); fits=[]
def chk(name,d): fits.append((score(d,y),name))
for i,j in [(i,j) for i in range(4) for j in range(i,4)]:
 for a in [t[i],t[0],t[1]]:
  for k in [t[j],t[0],t[1]]:
   gi=1/(1+a*q[:,i]**2); gj=1/(1+k*q[:,j]**2)
   for nm,g in [('i',gi),('j',gj),('prod',gi*gj),('sum',1/(1+a*q[:,i]**2+k*q[:,j]**2)),('sumplus',1/(1+(np.sqrt(a)*q[:,i]+np.sqrt(k)*q[:,j])**2)),('summinus',1/(1+(np.sqrt(a)*q[:,i]-np.sqrt(k)*q[:,j])**2)),('difference',1/(1+a*(q[:,i]-q[:,j])**2)),('couple',1/(1+a*(q[:,i]*q[:,j])**2))]:
    for tag,F in [('base',np.c_[o,c,b*g[:,None]]),('baseC',np.c_[o,(np.c_[b,c])*g[:,None]]),('inter',np.c_[o,c,b*g[:,None],b[:,3]*q[:,i]**3*g,b[:,3]*q[:,j]**3*g]),('addq',np.c_[o,c,b*g[:,None],q]),('split',np.c_[o,c,b*gi[:,None],b*gj[:,None]])]: chk((i,j,a,k,nm,tag),F)
for r in sorted(fits)[:24]: print(r)
