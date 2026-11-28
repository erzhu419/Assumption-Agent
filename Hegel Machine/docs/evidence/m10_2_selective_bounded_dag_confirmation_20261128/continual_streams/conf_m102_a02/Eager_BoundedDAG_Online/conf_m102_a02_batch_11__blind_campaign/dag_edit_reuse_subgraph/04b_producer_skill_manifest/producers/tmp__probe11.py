import numpy as np,pickle,importlib.util,itertools
from scipy.special import erf
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; n=52

def aff(x):return np.c_[np.ones(len(x)),x]
def fit(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def polys(a,k):return np.column_stack([np.prod(a[:,c],axis=1) if c else np.ones(len(a)) for j in range(k+1) for c in itertools.combinations_with_replacement(range(a.shape[1]),j)])
def ev(name,a,b=y):
 p=a@fit(a[:n],b[:n]); print(name,a.shape[1],np.mean(abs(b[:n]-p[:n])),np.mean(abs(b[n:]-p[n:])))
for k,p in ((k,s[k]) for k in ('panel_a','panel_b','panel_c')):
 u=p['u']; print(k)
 for name,v in p.items():
  if name=='u':continue
  print(name,'ranges',u.min(),u.max(),v.min(),v.max())
  for a in (aff(u),polys(u,2)):
   e=v-a@fit(a,v); print(a.shape[1],np.sqrt(np.mean(e**2)),np.quantile(np.linalg.norm(e,axis=1),[0,.25,.5,.75,1]));
  if k!='panel_a':
   def d(u,w):return np.c_[aff(u),erf(w*u)]
   f=lambda w:np.mean((v-d(u,w)@fit(d(u,w),v))**2)
   w=minimize_scalar(f,bounds=(.1,5),method='bounded').x; print('erf',w,np.sqrt(f(w)))
for j in (1,2,3):ev('x poly',polys(x,j));ev('traj x poly',polys(x,j),t)
sp=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/conf_m102_a02_batch_06_solver.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
z=old._calibrate(s['panel_a'],x);g=old._panel(s['panel_b'],z);v=z*np.roll(g,1,axis=1)
for name,a in [('z',z),('g',g),('base',np.c_[g,z,v]),('base+cross',np.c_[g,z,v,g[:,:2]*g[:,2:]])]:
 ev(name,aff(a));ev('traj '+name,aff(a),t)
 if a.shape[1]==4:ev(name+'quad',polys(a,2));ev('traj '+name+'quad',polys(a,2),t)
p,q=old._predict(z,g,np.arange(n),np.arange(n,len(x)),y[:n],t[:n]); print('old actual',np.mean(abs(p-y[n:])),np.mean(abs(q-t[n:])))
