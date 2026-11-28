import numpy as np, pickle
from itertools import combinations_with_replacement as cr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=130)
def design(x,d=1): return np.column_stack([np.ones(len(x))]+[np.prod(x[:,idx],axis=1) for n in range(1,d+1) for idx in cr(range(x.shape[1]),n)])
def hold(f,y,n=52):
 c=np.linalg.lstsq(f[:n],y[:n],rcond=None)[0]; return np.mean(abs(y[n:]-f[n:]@c))
print('times',s['eval_times'])
for p in ['panel_a','panel_b','panel_c']:
 u=s[p]['u']; print(p, 'u range',u.min(0),u.max(0)); print('u ex',u[:3]);
 for k,v in s[p].items():
  if k=='u':continue
  print(k,'v ex',v[:3]);
  for fun in [lambda x:x,np.tanh,np.sin,lambda x: x/(1+abs(x)),lambda x: x/(1+x*x)]:
   f=design(fun(u)); c=np.linalg.lstsq(f,v,rcond=None)[0]; r=np.linalg.norm(v-f@c,axis=1); keep=np.argsort(r)[:int(.88*len(r))]; c=np.linalg.lstsq(f[keep],v[keep],rcond=None)[0]; print(fun.__name__, 'r med,max',np.median(np.linalg.norm(v-f@c,axis=1)),np.max(r),'coef',c if fun.__name__=='<lambda>' and fun(u) is u else '')
  print('poly held', [hold(design(u,d),v,int(.75*len(u))) for d in (1,2)])
t=s['train']; x=t['x']; y=t['y']; z=t['traj']; print('poly CV',[(d,hold(design(x,d),y),hold(design(x,d),z)) for d in range(1,5)]);
print('affine y+traj',np.linalg.lstsq(design(x),np.c_[y,z[:,0],z[:,-1]],rcond=None)[0]);print('y poly',np.linalg.lstsq(design(x,2),y,rcond=None)[0]);
