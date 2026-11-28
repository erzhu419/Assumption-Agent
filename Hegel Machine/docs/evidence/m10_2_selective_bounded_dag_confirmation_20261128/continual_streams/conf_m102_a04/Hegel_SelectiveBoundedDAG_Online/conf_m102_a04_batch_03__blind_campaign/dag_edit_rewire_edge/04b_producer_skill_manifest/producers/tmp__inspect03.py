import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task/_frozen_nodes')
import node_00 as a,node_01 as b,node_02 as c
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; n=len(tr['y'])
np.set_printoptions(precision=4,suppress=True,linewidth=120)
print('times',t,'y range',np.quantile(tr['y'],[0,.25,.5,.75,1]))
for k in ('panel_a','panel_b','panel_c'):
 print(k,[(kk,v[:3]) for kk,v in s[k].items()]); print('align',a._align(s[k]) if 'v' in s[k] else '')
for seed in [-1,0,1,2]:
 idx=np.arange(n) if seed<0 else np.random.default_rng(seed).permutation(n); i,j=idx[:54],idx[54:]; ss={**s,'train':{k:tr[k][i] for k in ('x','y','traj')},'test':{'x':tr['x'][j]}}; W=a.run_node({'__input__':ss}); T=b.run_node({'__input__':ss}); y=T['targets']; z=np.c_[tr['y'][j],tr['traj'][j]]; base=c.run_node({'node_00':W,'node_01':T})['fit']; w=W['features']; A=T['basis']; P=np.linalg.pinv(A); coeff=y[:,1:]@P.T
 def err(p): return np.round(np.mean(abs(p-z[:,0])),6) if p.ndim==1 else np.round(np.mean(abs(p-z[:,1:])),6)
 print('fold',seed,'BASE',err(base[:,0]),err(base[:,1:]@P.T@A.T),'basis fit noise',np.mean(abs(coeff@A.T-y[:,1:])))
 off,sc=a._align(s['panel_a']); raw=(np.r_[tr['x'][i],tr['x'][j]]-off)/sc
 if seed==-1: print('features',w[:8], 'feature ranges',np.quantile(w,[0,.5,1],axis=0),'coeff ranges',np.quantile(coeff,[0,.5,1],axis=0),'corr with targets',np.corrcoef(np.c_[raw[:len(y)],w[:len(y)],y[:,0],coeff].T)[:-3,-3:])
 for name,v in [('raw',raw),('soft',w[:,:4]),('curve',w[:,4:]),('all',w),('raw+all',np.c_[raw,w])]:
  for d in (1,2,3):
   F=PolynomialFeatures(d).fit_transform(v); U,D,V=np.linalg.svd(F[:len(i)],full_matrices=False); K=F[len(i):]@V.T
   pairs=[]
   for r in (1e-8,.001,.01,.1,1):
    p=(K*(D/(D*D+r)))@(U.T@y); pairs.append((err(p[:,0]),err(p[:,1:]@P.T@A.T)))
   print(name,d,pairs)
