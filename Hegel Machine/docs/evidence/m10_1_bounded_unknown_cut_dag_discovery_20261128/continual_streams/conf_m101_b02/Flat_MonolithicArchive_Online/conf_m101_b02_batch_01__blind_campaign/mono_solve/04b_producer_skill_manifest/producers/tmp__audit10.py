import sys, pickle
import numpy as np
sys.path.insert(0, '/holdout_task')
from batch_solver import _nodes, _features, _odd
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['train']; x=p['x']; y=p['y']; t=p['traj']; Y=np.c_[y,t]; a,b,c=_nodes(s,x); o=_odd(b)
def f(z): return np.c_[a,z,(a[:,:,None]*z[:,None,:]).reshape(len(a),-1)]
def ls(g,v):return np.linalg.lstsq(g,v,rcond=None)[0]
G=f(c); d=np.c_[np.ones(len(x)),G]; w=ls(d,Y);print('cross coefficients scalar and trajectory modes per term');print(np.round(w[9:,0].reshape(4,4),3)); _,_,vt=np.linalg.svd(t,full_matrices=False);print(np.round((w[9:,1:]@vt[:2].T).reshape(4,4,2),3));print('full cross residual',np.mean(abs(d@w-Y),axis=0))
rng=np.random.default_rng(425); ix=np.arange(len(x)); splits=[('52/20',ix[:52],ix[52:]),('56/16',ix[:56],ix[56:])];splits += [('random',z[:52],z[52:]) for z in [rng.permutation(ix) for _ in range(30)]]; order=np.argsort(np.linalg.norm(a,axis=1)); splits.append(('high-radius',order[:56],order[56:]));order=np.argsort(np.max(abs(a),axis=1));splits.append(('high-coordinate',order[:56],order[56:])); order=np.argsort(np.sum(abs(a)>.35,axis=1));splits.append(('low-active',order[16:],order[:16]))
for name,gs,gt,reg,rank in [('baseline',_features(a,c),_features(a,c),.015,8),('cross',G,G,0,8),('cross rank2',G,G,0,2),('diag/cross rank2',np.c_[a,c,a*c],G,0,2),('diag/cross rank2 ridge',np.c_[a,c,a*c],G,.015,2),('odd cross rank2',f(o),f(o),0,2)]:
 def predict(g,tar,i,j):
  z=(g-g[i].mean(0))/np.maximum(g[i].std(0),1e-12);v=tar-tar[i].mean(0); q=z[i];w=ls(q,v[i]) if not reg else np.linalg.solve(q.T@q+reg*np.eye(q.shape[1]),q.T@v[i]);return z[j]@w+tar[i].mean(0)
 out=[]
 for label,i,j in splits:
  pp=predict(gt,t,i,j);tp=np.linalg.svd(t[i],full_matrices=False)[2][:rank];e=abs((pp@tp.T)@tp-t[j]);out.append((label,np.mean(abs(predict(gs,y,i,j)-y[j])),e.mean(),e.mean(axis=0).max()))
 r=np.array([q[1:] for q in out if q[0]=='random']); print(name,'ordered/context',[(q[0],np.round(q[1:],5).tolist()) for q in out if q[0]!='random'],'random worst',np.round(r.max(0),5))
