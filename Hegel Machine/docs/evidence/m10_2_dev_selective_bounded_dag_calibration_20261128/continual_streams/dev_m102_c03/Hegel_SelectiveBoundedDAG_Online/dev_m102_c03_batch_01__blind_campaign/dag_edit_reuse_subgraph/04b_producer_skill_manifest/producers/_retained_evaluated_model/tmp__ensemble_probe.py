exec(open('/tmp/readout_probe.py').read().split('res=[]')[0]); mod=runpy.run_path('/holdout_task/node_01.py'); P=[]
from sklearn.linear_model import ARDRegression
from itertools import combinations
for a,b,base in folds:
 xx=x[np.r_[a,b]]; n=len(a); z=mod['_latent_coordinates'](st['panel_a'],xx); A,Q=mod['_mode_readout'](st['eval_times'],Y[a,1:]); T=np.c_[Y[a,0],A]; opts=[]
 for mix in (1.,0.,.5):
  s,g=mod['_features'](mix*z+(1-mix)*xx); zz=mix*z+(1-mix)*xx
  f=np.column_stack(s+g+[np.prod(zz[:,j],axis=1) for j in combinations(range(4),3)])[:,1:]; f=(f-f[:n].mean(0))/np.maximum(f[:n].std(0),1e-9)
  for k in (10000,20000,30000,50000,100000):
   m=np.column_stack([ARDRegression(threshold_lambda=k).fit(f[:n],t).predict(f[n:]) for t in T.T]);opts.append(np.c_[m[:,0],m[:,1:]@Q.T]);
 P.append((np.array(opts),Y[b]))
G={'base':[2], 'cal':[0,1,2,3,4], 'calmid':[1,2,3], 'all':list(range(15)), 'allmid':[i for i in range(15) if i%5 in (1,2,3)], 'calmild':[0,1,2], 'calmidblend':[1,2,3,11,12,13], 'calplusraw':[0,1,2,3,4,5,6,7,8,9]}
for key,inds in G.items():
 out=[]
 for p,y in P:
  d=abs(p[inds].mean(0)-y);out.append([d[:,0].mean(),d[:,1:].mean(),d[:,1:].mean(0).max()]);
 print(key,np.round(out,5).tolist(),'max',np.max(out,0).round(5))
