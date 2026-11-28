import numpy as np,pickle,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(s['panel_a'][k] for k in ['u','v']); tr=s['train']; print('train x range',tr['x'].min(0),tr['x'].max(0)); print('panel b/c samples');
for key in ('panel_b','panel_c'):
 p=s[key];print(key, np.round(np.column_stack(tuple(p.values()))[:4],4));
good=np.max(abs(v),axis=1)<4; print('suspect rows',np.where(~good)[0]); u,v=u[good],v[good];
for k in (1,2,3):
 terms=[()]+[a for d in range(1,k+1) for a in itertools.combinations_with_replacement(range(4),d)]; X=np.column_stack([np.prod(u[:,a],axis=1) for a in terms]); c=np.linalg.lstsq(X,v,rcond=None)[0]; r=v-X@c; h=np.sum(X*np.linalg.pinv(X).T,axis=1); print('degree',k,'rmse',np.mean(r*r,0)**.5,'LOO MAE',np.mean(abs(r/(1-h[:,None])),0));
 for j in range(4): print(j,[(a,round(c[i,j],5)) for i,a in enumerate(terms) if abs(c[i,j])>0.025]);
print('clean rows');print(np.round(np.c_[u,v],5))
