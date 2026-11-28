import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; y=t['y']; tr=t['traj']; tm=s['eval_times']; np.set_printoptions(precision=5,suppress=True)
def affine(x):return np.c_[np.ones(len(x)),x]
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,i],axis=1) for j in range(1,d+1) for i in combinations_with_replacement(range(x.shape[1]),j)])
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def fit(a,b):return a@ls(a,b)
p=s['panel_a']; a=affine(p['u']); z=p['v']; good=np.linalg.norm(z-fit(a,z),axis=1)<3; A=ls(a[good],z[good]); print('A',A,'resid RMS',np.sqrt(np.mean((a[good]@A-z[good])**2)),'bad',np.where(~good)[0]);
for k in ['panel_a','panel_b','panel_c']:
 p=s[k]; u=p['u']; g=good if k=='panel_a' else slice(None); u=u[g];
 for name,z in p.items():
  if name=='u':continue
  z=z[g]; print(k,name,'range',z.min(0),z.max(0),'resid svals',np.linalg.svd(z-fit(affine(u),z),compute_uv=False));
  for d in [1,2,3]:
   f=poly(u,d); err=[]
   for i in range(len(u)):
    ii=np.arange(len(u))!=i;err.append(f[i]@ls(f[ii],z[ii])-z[i]);
   print(d,'train',np.mean(abs(fit(f,z)-z),axis=0),'loo',np.mean(np.abs(err),axis=0));
U,S,V=np.linalg.svd(tr,full_matrices=False); v=V[0]/V[0,-1]; h=tr@v/(v@v)
print('times',tm,'telemetry',s['telemetry_vec'],'trajectory basis',v,'amplitude range',h.min(),h.max(),'y range',y.min(),y.max()); print('h/y samples',np.c_[h[:10],y[:10]]);
for name,u in [('x',x),('A',affine(x)@A),('invA',(x-A[0])@np.linalg.inv(A[1:]))]:
 for d in [1,2,3]:
  f=poly(u,d); c=ls(f[:52],np.c_[y,h][:52]); print(name,d,'CV MAE',np.mean(abs(f[52:]@c-np.c_[y,h][52:]),axis=0));
for d in [1,2,3,4]:
 f=poly(h[:,None],d); print('y from amp',d,'mae',np.mean(abs(f[52:]@ls(f[:52],y[:52])-y[52:])))
for target in [h,y]:
 w=ls(affine(x[:52]),target[:52]);
 for name,fun in [('tanh',np.tanh),('sin',np.sin),('exp',lambda v:np.exp(np.clip(v,-30,30)))]:
  def pred(p):return p[0]+p[1]*fun(affine(x)@p[2:]);
  p=least_squares(lambda p:(pred(p)-target)[:52],np.r_[0,1,w],max_nfev=1000).x; print(name,'fit',np.mean(abs((pred(p)-target)[:52])),'cv',np.mean(abs((pred(p)-target)[52:])),'params',p)
