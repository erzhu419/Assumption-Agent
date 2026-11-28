import sys,pickle,itertools,importlib.util
import numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0,'/holdout_task')
import node_00 as cal
s0=importlib.util.spec_from_file_location('head','/holdout_task/_frozen_nodes/node_01.py'); h=importlib.util.module_from_spec(s0);s0.loader.exec_module(h)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; X=t['x']; y=t['y']; traj=t['traj']; n=len(X); w=cal.run_node({'__input__':s})['w'][:n]
print('n',n,'w min max mean std',w.min(0),w.max(0),w.mean(0),w.std(0)); print('panel-B V',s['panel_b']['v'].min(0),s['panel_b']['v'].max(0),s['panel_b']['v'].mean(0))
rng=np.random.default_rng(78414); folds=[(np.arange(n-20),np.arange(n-20,n))];
for b in np.array_split(rng.permutation(n),5): folds.append((np.setdiff1d(np.arange(n),b),b))
def one(W, aa, bb, v, interaction, kind, bounds=(.8,7)):
 wa=W[aa]; vb=v[aa]; scale=np.maximum(np.std(vb,axis=0),1e-8)
 def err(g):
  f=h._features(wa,g,interaction,kind); return np.mean(((f@h._ls(f,vb)-vb)/scale)**2)
 g=minimize_scalar(err,bounds=bounds,method='bounded').x
 return h._features(W[bb],g,interaction,kind)@h._ls(h._features(wa,g,interaction,kind),vb),g,err(g)**.5

def check(W, verbose=False):
 stats=[]
 for a,b in folds:
  ya,ga,ea=one(W,a,b,y,True,'asinh'); yt,gt,et=one(W,a,b,y,True,'atan'); p,gp,ep=one(W,a,b,traj,False,'inv3'); m=lambda z:np.mean(abs(z-y[b])); stats.append([m(.35*ya+.65*yt),np.mean(abs(p-traj[b])),m(ya),m(yt),ga,gt,gp,ea,et,ep]);
 st=np.array(stats)
 if verbose: print(st)
 return np.round(np.r_[st[0,:2],st[1:,:2].max(0),st[1:,:2].mean(0)],6)
np.set_printoptions(precision=5,suppress=True)
print('baseline',check(w,True))
for fac in [.125,.25,.5,.75,1,1.5,2,3,5,8]: print('scale',fac,check(w*fac, fac in [.25,.5,2]))
for mix in [-1.,-.5,-.25,-.125,.125,.25,.5,1.]: print('center',mix,check(w+mix*w.mean(0)))
for p in [.25,.5,.75,1.25,1.5,2.,3.]: print('power',p,check(np.sign(w)*abs(w)**p))
