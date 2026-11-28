import pickle, numpy as np
from itertools import combinations_with_replacement
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=p['train']['x']; y=p['train']['y']; t=p['train']['traj']; a=np.c_[np.ones(len(x)),x]; n=len(x)
def poly(x,d): return np.column_stack([np.prod(x[:,ix],axis=1) for j in range(d+1) for ix in combinations_with_replacement(range(x.shape[1]),j)])
def report(F,Y=y):
 b=np.linalg.lstsq(F,Y,rcond=None)[0]; e=Y-F@b; loo=[]
 for i in range(6):
  v=np.arange(n)%6==i; loo.append(np.mean(abs(Y[v]-F[v]@np.linalg.lstsq(F[~v],Y[~v],rcond=None)[0])))
 print('in',np.mean(abs(e)),'blocked',np.mean(loo),max(loo),'rank',np.linalg.matrix_rank(F)); return b,e
for d in range(1,6):
 print('degree',d); b,e=report(poly(x,d)); print(np.round(b,4) if d<=3 else '')
print('trajectory'); c,e=report(a,t); print('coef',np.round(c,4),'resid sv',np.linalg.svd(e,compute_uv=False),'fit sv',np.linalg.svd(a@c,compute_uv=False)); print('end vs final state',np.corrcoef(y,t[:,-1])[0,1])
print('training samples'); print(np.round(np.c_[x,t[:,-1],y],3))
for k in ['panel_a','panel_b','panel_c']:
 print(k)
 for j,v in p[k].items():
  print(j,'mu',np.round(v.mean(0),3),'std',np.round(v.std(0),3));
  if j!='u': report0=np.linalg.lstsq(poly(p[k]['u'],1),v,rcond=None)[0]; print(np.round(report0,3), 'MAE',np.mean(abs(poly(p[k]['u'],1)@report0-v)))
