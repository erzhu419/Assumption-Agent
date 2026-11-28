import pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_a']['u']; v=s['panel_a']['v']; X=np.c_[np.ones(len(u)),u]; ls=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]
np.set_printoptions(precision=5,suppress=True,linewidth=150)
R=v-X@ls(X,v); a,d,b=np.linalg.svd(R,full_matrices=False); print('residual svd',d,b); print('u,v,top residual:', np.c_[u,v,a[:,0]*d[0]])
g=np.random.default_rng(119); best=None
for kind in ('full','diag'):
 best=(np.inf,None)
 for t in range(3000):
  j=g.choice(len(u),5,replace=False)
  if kind=='full': B=ls(X[j],v[j]); P=X@B
  else:
   B=np.array([ls(X[j][:,[0,i+1]],v[j,i]) for i in range(4)]); P=B[:,0]+u*B[:,1]
  err=np.linalg.norm(P-v,axis=1); score=np.median(err)
  if score<best[0]:best=(score,B)
 B=best[1]; P=X@B if kind=='full' else B[:,0]+u*B[:,1]; print(kind,best);print('row errors',np.sort(np.linalg.norm(P-v,axis=1))); print('percol quantiles',np.quantile(abs(P-v),[0,.25,.5,.7,.8,.9,1],axis=0))
