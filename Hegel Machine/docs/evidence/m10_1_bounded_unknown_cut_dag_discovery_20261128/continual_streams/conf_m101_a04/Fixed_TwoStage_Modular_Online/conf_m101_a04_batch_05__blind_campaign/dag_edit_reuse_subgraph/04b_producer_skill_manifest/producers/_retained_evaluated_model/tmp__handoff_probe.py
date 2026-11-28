import copy, pickle, sys
import numpy as np
from scipy.optimize import minimize_scalar
sys.path[:0] = ['/holdout_task', '/holdout_task/_frozen_nodes']
import node_00 as m, node_01 as f
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
z,b,c1,c2 = m._features(s)
p = s['panel_c']; u,v = p['u'],p['v2']
def smooth(x,a):
    return x*x/(np.sqrt(1+a*x*x)+1)
def fit(d,y):
    val = minimize_scalar(lambda a: np.mean((m._design(d,smooth(d,a)) @ m._fit(m._design(d,smooth(d,a)),y)-y)**2),bounds=(.01,3),method='bounded').x
    return val,m._fit(m._design(d,smooth(d,val)),y)
err=[]
for i in range(len(u)):
    mask=np.arange(len(u))!=i; a,c=fit(u[mask],v[mask]); err.append((m._design(u[i:i+1],smooth(u[i:i+1],a))@c-v[i])[0])
a,c=fit(u,v); improved=m._design(z,smooth(z,a))@c
print('v2 nested LOO RMSE by output',np.sqrt(np.mean(np.array(err)**2,axis=0)),'shape',a)
t=s['train']; y,st=t['y'],t['traj']; n=len(y); ix=np.arange(n)
blocks=[ix[-20:]]
for seed in (52,931,1901): blocks.extend(np.array_split(np.random.default_rng(seed).permutation(n),5))
blocks.extend(np.array_split(ix[np.linalg.norm(z[:n],axis=1).argsort()],5))
def score(h):
    out=[]; worst_time=[]
    for te in blocks:
        tr=np.setdiff1d(ix,te)
        pred=f.run_node({'node_00': {'h':np.r_[h[tr],h[te]],'n_train':np.asarray(len(tr)),'y':y[tr],'traj':st[tr]}})
        e=np.abs(pred['state_pred']-st[te]); out.append((np.mean(abs(pred['y_hat']-y[te])),np.mean(e))); worst_time.append(e.mean(0).max())
    out=np.asarray(out); return np.round(out[0],6),np.round(out.max(0),6),np.round(out[1:16].mean(0),6),np.round(out[16:].max(0),6),round(max(worst_time),6)
for name, h in [('original',np.c_[z,b,c1,c2]),('c-only cosine',np.c_[c1,c2]),('c-only sqrt',np.c_[c1,improved])]+[(f'b-scale {w}',np.c_[w*b,c1,improved]) for w in (.03,.1,.2,.35,.5,1)]+[(f'z+b scale {w}',np.c_[w*z,w*b,c1,improved]) for w in (.05,.12,.25,.5)]:
    print(name, score(h))
