import numpy as np,pickle,glob,importlib.util,warnings
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; n=52
tr=s['train']; yc=tr['y'][n:]; tc=tr['traj'][n:]; sc={**s,'train':{k:v[:n] if k!='summary' else v for k,v in tr.items()},'test':{'x':tr['x'][n:]}}
def aff(x):return np.c_[np.ones(len(x)),x]
def fit(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
for file in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 try:
  with warnings.catch_warnings():
   warnings.simplefilter('ignore'); spec=importlib.util.spec_from_file_location('m',file); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); o=m.solve_batch(sc)
  print(file.split('/')[-1],np.mean(abs(yc-o['y_hat'])),np.mean(abs(tc-o['state_pred'])),flush=True)
 except Exception as e:print(type(e).__name__,str(e),flush=True)
for pan,keys in [('panel_b',['v']),('panel_c',['v1','v2'])]:
 for key in keys:
  u,v=s[pan]['u'],s[pan][key]; out=[]
  for name,f,p,lo,hi in [('lin',lambda x,p:x,[1],[0],[2]),('tanh',lambda x,p:np.tanh(p[0]*x+p[1]),[1,0],[.01,-4],[8,4]),('cos',lambda x,p:np.cos(p[0]*x+p[1]),[2.2,0],[.1,-4],[6,4]),('soft',lambda x,p:np.logaddexp(0,p[0]*x+p[1]),[2.2,0],[.05,-5],[8,5]),('threshold',lambda x,p:np.sign(x)*np.maximum(abs(x)-p[0],0)**p[1],[.3,1.5],[0,.2],[.85,5]),('rat',lambda x,p:x/(1+p[0]*abs(x)**p[1]),[.5,2],[0,.2],[8,6]),('atan',lambda x,p:x+p[0]*np.arctan(p[1]*x),[.3,2],[0,.05],[3,8])]:
   def err(p):
    a=aff(f(u,p));return (a@fit(a,v)-v).ravel()
   for pp in ([p,[4,0],[1,0]] if name=='cos' else [p]):
    o=least_squares(err,pp);out.append((np.std(o.fun),name,o.x))
  print(pan,key,sorted(out,key=lambda q:q[0])[:8])
t=s['eval_times']; Y=tr['traj'][:n]; exp=lambda r:-np.expm1(-r*t)
for label,b,p in [('linear+exp',lambda p:np.array([t,exp(p[0])]),[1.5]),('two exp',lambda p:np.array([exp(p[0]),exp(p[1])]),[.4,1.5]),('exp two + constant',lambda p:np.array([np.ones_like(t),exp(p[0]),exp(p[1])]),[.4,1.5])]:
 def er(p):
  F=b(p);return (F.T@fit(F.T,Y.T)-Y.T).ravel()
 o=least_squares(er,p,bounds=(.001,15)); print(label,o.x,np.std(o.fun))
