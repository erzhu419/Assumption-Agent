import pickle,numpy as np
from itertools import combinations_with_replacement
np.set_printoptions(precision=5,suppress=True,linewidth=145)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['eval_times'];tr=s['train'];x=tr['x']; j=tr['traj']; y=tr['y']; ls=lambda X,Y:np.linalg.lstsq(X,Y,rcond=None)[0]; aff=lambda x:np.c_[np.ones(len(x)),x]
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,k],axis=1) for i in range(1,d+1) for k in combinations_with_replacement(range(x.shape[1]),i)])
ops={}
for k in ['panel_a','panel_b','panel_c']:
 p=s[k];u=p['u']; ii=np.arange(len(u)); A=aff(u)
 for name,v in p.items():
  if name=='u':continue
  if k=='panel_a':
   for _ in range(4):ii=np.argsort(np.linalg.norm(A@ls(A[ii],v[ii])-v,axis=1))[:45]
  b=ls(A[ii],v[ii]); ops[k+name]=b;print(k,name,'affine residual quantiles',np.quantile(np.linalg.norm(A@b-v,axis=1),[0,.5,.9,.94,1]),'coef',b)
  Q=poly(u,2);coef=ls(Q[ii],v[ii]);print('quadratic RMS',np.sqrt(np.mean((Q[ii]@coef-v[ii])**2)), 'diag/factor',coef[5:])
print('times',t,'summary',tr['summary'],'y distributions',np.quantile(y,[0,.1,.5,.9,1]));print('ranges x',x.min(0),x.max(0),'test',s['test']['x'].min(0),s['test']['x'].max(0));print('first rows x/y/trajectory',np.c_[x,y,j][:10]);print('sv raw center diff',*[np.linalg.svd(a,compute_uv=False) for a in (j,j-j.mean(0),np.diff(j,axis=1))]);print('y corr',np.corrcoef(y,j.T)[0]);print('y from traj MAE',np.mean(abs(aff(j)@ls(aff(j)[:52],y[:52])-y)),ls(aff(j)[:52],y[:52]));
for name,z in [('raw',x)]+[(k,aff(x)@v) for k,v in ops.items()]+[('A->B',aff(aff(x)@ops['panel_av'])@ops['panel_bv'])]:
 print(name,'stats',z.mean(0),z.std(0));
 for d in [1,2,3]:
  X=poly(z,d); P=X[52:]@ls(X[:52],np.c_[y,j][:52]);print('degree',d,'mae',np.mean(abs(P-np.c_[y,j][52:]),axis=0))
