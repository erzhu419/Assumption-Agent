import pickle,numpy as np,sys,importlib.util
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; n=52
p='/tmp/hegel_continual_archive/conf_m102_a01_batch_03_solver.py'; q=importlib.util.spec_from_file_location('old',p); m=importlib.util.module_from_spec(q); q.loader.exec_module(m)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
np.set_printoptions(precision=5,suppress=True,linewidth=160)
z,h=m._coordinates(s,x); C=m._panel_coordinates(s,x)
for p in ['panel_a','panel_b','panel_c']:
 u=s[p]['u']; a=m._affine(u)
 for k,v in s[p].items():
  if k=='u':continue
  f=ls(a,v); r=v-a@f; print(p,k,'affine',f,'err',np.quantile(abs(r),[0,.5,.9,1]))
  for fun in [lambda u,j:u/(1+j*abs(u)),lambda u,j:np.sqrt(1+j*u*u),lambda u,j:np.tanh(j*u),lambda u,j:u/(1+j*u*u),lambda u,j:u**2,lambda u,j:u**3]:
   def fun2(j):
    b=np.c_[a,fun(u,j)]; return np.mean((v-b@ls(b,v))**2)**.5
   j=minimize_scalar(fun2,bounds=(.01,4),method='bounded').x; print(round(j,3),round(fun2(j),7),end='; ')
  print()
U,S,V=np.linalg.svd(t,full_matrices=False); a=t@V[0]; print('trajectory singular',S,'basis',V[:2]); print('corr',np.corrcoef(np.c_[x,z,h,y,a].T)[-2:]);
for i in np.argsort(abs(a))[:18]:print('low',i,x[i],z[i],h[i],y[i],a[i])
for i,w in enumerate(C):
 b=m._affine(w); print('coord',i,[(np.mean(abs(r[n:]-A[n:]@ls(A[:n],r[:n]))),np.mean(abs(r[:n]-A[:n]@ls(A[:n],r[:n])))) for r in [y,a] for A in [b,np.c_[b,m._affine(x)[:,1:],a] if r is y else b,np.c_[b,*[w[:,j]*w[:,k] for j in range(4) for k in range(j,4)]]]])
cv={**s,'train':{**s['train'],**{k:s['train'][k][:n] for k in ['x','y','traj']}},'test':{'x':x[n:]}}; o=m.solve_batch(cv); print('arch03',np.mean(abs(y[n:]-o['y_hat'])),np.mean(abs(t[n:]-o['state_pred']))); print(np.c_[x[n:],y[n:],o['y_hat'],y[n:]-o['y_hat'],t[n:,-1],o['state_pred'][:,-1]])
