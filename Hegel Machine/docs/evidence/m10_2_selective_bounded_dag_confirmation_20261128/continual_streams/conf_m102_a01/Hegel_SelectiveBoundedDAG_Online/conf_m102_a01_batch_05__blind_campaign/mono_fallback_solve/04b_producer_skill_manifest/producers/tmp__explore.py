import numpy as np,pickle,sys
from itertools import combinations_with_replacement
sys.path.insert(0,'/tmp/hegel_continual_archive')
import conf_m102_a01_batch_03_solver as old
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];t=s['train'];x=t['x'];z,h=old._coordinates(s,x);f=lambda x:np.c_[np.ones(len(x)),x];b=np.linalg.lstsq(f(s['panel_b']['u']),s['panel_b']['v'],rcond=None)[0];q=f(z)@b
np.set_printoptions(precision=3,suppress=True,linewidth=150)
def terms(x,d):
 ids=[i for k in range(d+1) for i in combinations_with_replacement(range(4),k)];return np.column_stack([np.prod(x[:,i],axis=1) for i in ids]),ids
def inspect(x,y):
 for d in [1,2,3]:
  a,ids=terms(x,d);w=np.linalg.lstsq(a,y,rcond=None)[0];print('deg',d,'rmse',np.sqrt(np.mean((y-a@w)**2,axis=0)),'loo',np.mean(abs((y-a@w)/np.maximum(1e-9,1-(np.linalg.qr(a)[0]**2).sum(1))[:,None]),0) if a.shape[1]<len(a) else '-');
  if d==2: print(list(zip(ids,w.round(3))))
p=s['panel_c']; print('C raw');inspect(p['u'],np.c_[p['v1'],p['v2']]); print('C arctanh max',abs(p['v1']).max(0),abs(p['v2']).max(0)); print('C conditioned on v1');inspect(p['v1'],p['v2']);
u,sv,vt=np.linalg.svd(t['traj'],full_matrices=False);print('shapes',vt[:3]);y=np.c_[t['y'],u[:,:2]*sv[:2]]
for name,g in [('x',x),('z',z),('h',h),('q',q)]:
 a,ids=terms(g,3);print(name,'ranges',g.min(0),g.max(0));
 for p in [5,15,35]:
  w=np.linalg.lstsq(a[:52,:p],y[:52],rcond=None)[0]; print(p, np.mean(abs(a[52:,:p]@w-y[52:]),0));
 if name in ['z','h']: print('quad coefficients',list(zip(ids[:15],np.linalg.lstsq(a[:,:15],y,rcond=None)[0].round(3))))
