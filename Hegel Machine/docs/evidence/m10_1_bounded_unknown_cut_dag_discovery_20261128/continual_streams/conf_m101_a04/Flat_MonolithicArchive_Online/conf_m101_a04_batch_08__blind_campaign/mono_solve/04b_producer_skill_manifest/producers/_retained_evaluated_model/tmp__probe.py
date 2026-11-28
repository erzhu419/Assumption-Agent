import numpy as np,pickle,itertools,glob,importlib.util
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; n=52
bias=lambda u:np.c_[np.ones(len(u)),u]
def poly(u,d): return np.column_stack([np.prod(u[:,p],axis=1) for j in range(d+1) for p in itertools.combinations_with_replacement(range(u.shape[1]),j)])
def fit(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
for pn in ('panel_a','panel_b','panel_c'):
 p=s[pn];u=p['u'];
 for k,v in p.items():
  if k=='u':continue
  print(pn,k)
  for d in (1,2,3):
   f=poly(u,d); pred=np.array([f[i]@fit(np.delete(f,i,0),np.delete(v,i,0)) for i in range(len(u))]);print('poly',d,np.sqrt(np.mean((pred-v)**2,axis=0)))
  for name,fn in [('sin',lambda u,w:np.sin(w*u)),('cos',lambda u,w:np.cos(w*u)),('sincos',lambda u,w:np.c_[np.sin(w*u),np.cos(w*u)]),('tanh',lambda u,w:np.tanh(w*u)),('rational',lambda u,w:u/(1+w*u*u)),('sin2',lambda u,w:np.c_[np.sin(w*u),np.sin(2*w*u)])]:
   for base in (False,True):
    def calc(w):
     f=bias(np.c_[u,fn(u,w)] if base else fn(u,w));return np.mean((f@fit(f,v)-v)**2)
    ws=np.linspace(.1,8,80);i=np.argmin([calc(w) for w in ws]);r=minimize_scalar(calc,bounds=(ws[max(i-1,0)],ws[min(i+1,79)]),method='bounded'); print(name,base,round(r.x,5),round(r.fun**.5,6))
  if pn=='panel_a':
   mask=np.ones(len(u),bool)
   for _ in range(8):
    coef=fit(bias(u[mask]),v[mask]);e=np.linalg.norm(bias(u)@coef-v,axis=1);mask=e<max(5*np.median(e),.0006)
   print('robust',coef,'inlier count',sum(mask),'median',np.median(e));
t=s['train']; ss=dict(s,train={k:v[:n] if k!='summary' else v for k,v in t.items()},test={'x':t['x'][n:]});
for fn in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 spec=importlib.util.spec_from_file_location('a',fn);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 try:
  p=m.solve_batch(ss);print(fn[-12:],np.mean(abs(p['y_hat']-t['y'][n:])),np.mean(abs(p['state_pred']-t['traj'][n:])))
 except Exception as e:print(str(e))
print('traj sv',np.linalg.svd(t['traj'],compute_uv=False))
