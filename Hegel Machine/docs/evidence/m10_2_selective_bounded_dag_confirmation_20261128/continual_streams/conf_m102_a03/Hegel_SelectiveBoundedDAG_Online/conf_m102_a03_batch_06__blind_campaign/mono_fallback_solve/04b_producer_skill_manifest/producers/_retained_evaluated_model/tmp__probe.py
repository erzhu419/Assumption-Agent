import pickle,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=150)
def pol(x,d): return np.column_stack([np.prod(x[:,j],axis=1) for k in range(d+1) for j in combinations_with_replacement(range(x.shape[1]),k)])
def fit(x,y):return np.linalg.lstsq(x,y,rcond=1e-11)[0]
def panel(p,k):
 u,v=p['u'],p[k]; a=pol(u,1); c=fit(a,v); r=np.linalg.norm(v-a@c,axis=1); keep=np.argsort(r)[:int(.85*len(r))] if len(r)==48 else np.arange(len(r)); c=fit(a[keep],v[keep]); print(k,'affine',c,'err',np.sort(np.linalg.norm(v-a@c,axis=1))); print('quad coef',fit(pol(u,2)[keep],v[keep]).T,'quad rms',np.std((v-pol(u,2)@fit(pol(u,2)[keep],v[keep]))[keep],axis=0))
if __name__=='__main__':
 print('times',s['eval_times']); print('singular',np.linalg.svd(s['train']['traj'],compute_uv=False));
 for n in 'abc':
  for k in s['panel_'+n]:
   if k!='u': panel(s['panel_'+n],k)
 x,y,t=[s['train'][k] for k in ('x','y','traj')];
 for d in (1,2,3,4):
  a=pol(x,d); p=a@fit(a[:52],np.column_stack((y,t))[:52]); print(d, np.mean(abs(p[:52]-np.column_stack((y,t))[:52]),axis=0),np.mean(abs(p[52:]-np.column_stack((y,t))[52:]),axis=0))
