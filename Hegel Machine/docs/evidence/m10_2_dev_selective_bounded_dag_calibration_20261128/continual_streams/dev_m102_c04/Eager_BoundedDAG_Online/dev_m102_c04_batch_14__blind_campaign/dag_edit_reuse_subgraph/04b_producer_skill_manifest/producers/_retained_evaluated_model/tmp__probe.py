import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=5,suppress=True)
def fit(a,v):
 w=np.linalg.lstsq(a,v,rcond=1e-10)[0];e=np.linalg.norm(a@w-v,axis=1);return w,e
for k in ('panel_a','panel_b','panel_c'):
 u=s[k]['u']; a=np.c_[np.ones(len(u)),u]; print(k)
 for l,v in s[k].items():
  if l=='u': continue
  w,e=fit(a,v); print(l,'affine coefficients',w,'affine residual sorted',np.sort(e))
  if k=='panel_a':
   for j in range(4):
    ix=np.argsort(u[:,j]); print('coordinate',j,np.round(np.c_[u[:,j],v[:,j],(v-a@w)[:,j]][ix][::3],4))
  for name,f in [('square',lambda x,c:x*x),('cube',lambda x,c:x**3),('soft',lambda x,c:x/(1+c*abs(x))),('sat',lambda x,c:x*x/(1+c*x*x)),('tanh',lambda x,c:np.tanh(c*x)),('sin',lambda x,c:np.sin(c*x)),('xabs',lambda x,c:x*abs(x)),('odd_rat',lambda x,c:x/(1+c*x*x)),('exp',lambda x,c:np.exp(-c*x*x))]:
   def res(c):return fit(np.c_[a,f(u,c[0])],v)[1]
   o=least_squares(res,[.6],bounds=(.001,5.),max_nfev=60); print(name,round(o.x[0],5),'rms',np.mean(o.fun**2)**.5,'median',np.median(o.fun))
print('trajectory svd',np.linalg.svd(s['train']['traj'],compute_uv=False))
