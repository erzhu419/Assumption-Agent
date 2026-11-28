import glob, importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; l=np.linalg.lstsq
np.set_printoptions(precision=5,suppress=True,linewidth=145)
def affine(x):return np.c_[np.ones(len(x)),x]
def fit(x,y):return l(x,y,rcond=None)[0]
def score(a,Y,n=52):
 b=fit(a[:n],Y[:n]); return np.mean(abs(a[n:]@b-Y[n:])),np.mean(abs(a[:n]@b-Y[:n]))
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 sp=importlib.util.spec_from_file_location('p',f); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 try:
  st=dict(s,train={k:v[:52] if k!='summary' else v for k,v in tr.items()},test={'x':tr['x'][52:]});o=m.solve_batch(st);print(f[-19:],*(np.mean(abs(o[k]-tr[j][52:])) for k,j in [('y_hat','y'),('state_pred','traj')]))
 except Exception as e:print(e)
 if '02_solver' in f: mod=m
u,v=(s['panel_a'][k] for k in ('u','v'));c=mod._calibration(s['panel_a']);print('cal',c);z=mod._decode(tr['x'],c);print('cal first',z[:5]);print('sv trajectory',np.linalg.svd(tr['traj'][:52],compute_uv=False));
ops={}
for panel,name in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[panel]['u'],s[panel][name]
 def warp(x,p,kind):
  if kind=='threshold':return np.sign(x)*np.maximum(abs(x)-p[0],0)**p[1]
  if kind=='tanh':return np.tanh(x*p[0])
  if kind=='asinh':return np.arcsinh(x*p[0])
  if kind=='sin':return np.sin(x*p[0])
  if kind=='rational':return x/(1+p[0]*x*x)
  if kind=='power':return np.sign(x)*abs(x)**p[0]
  if kind=='clip':return np.clip(x,-p[0],p[0])
 for kind in ['threshold','tanh','asinh','sin','rational','power','clip']:
  def res(p):
   a=affine(warp(u,p,kind)); return (a@fit(a,v)-v).ravel()
  p=least_squares(res,[.35,1.6] if kind=='threshold' else [.8],bounds=([0,.4],[1.25,4]) if kind=='threshold' else ([.01],[6.])).x
  a=affine(warp(u,p,kind));b=fit(a,v);err=np.mean(abs(res(p)));print(name,kind,p,err)
  if name not in ops or err<ops[name][0]:ops[name]=(err,kind,p,b)
print('scores')
w={k:affine(warp(z,p,kind))@b for k,(_,kind,p,b) in ops.items()}
for k,x in dict(raw=tr['x'],z=z,**w).items():
 for a in [affine(x),np.c_[affine(x),*[x[:,i]*x[:,j] for i in range(4) for j in range(i,4)]]]:print(k,a.shape,score(a,tr['y']),score(a,tr['traj']))
for k,(_,kind,p,b) in ops.items():print('operator',k,kind,p,b)
