import sys,pickle,importlib.util
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar,least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def af(x):return np.c_[np.ones(len(x)),x]
def quad(x):return np.column_stack([af(x)]+[x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(x.shape[1]),2)])
def rm(x):return np.mean(x*x)**.5
for p in sorted(Path('/tmp/hegel_continual_archive').glob('*solver.py')):
 spec=importlib.util.spec_from_file_location('arch',p); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 if '09' in p.stem: old=m
 tr=s['train']; st={**s,'train':{**tr,**{k:tr[k][:52] for k in ('x','y','traj')}},'test':{'x':tr['x'][52:]}}
 try:
  o=m.solve_batch(st); print(p.stem,*[np.mean(abs(o[a]-tr[b][52:])) for a,b in [('y_hat','y'),('state_pred','traj')]])
 except Exception as e: print(e)
u,v=[s['panel_a'][k] for k in ('u','v')]; keep=np.linalg.norm(v-u,axis=1)<3
print('cal retained',keep.sum()); cal=old._calibration(s['panel_a']); print('cal09',cal)
z=old._invert(s['train']['x'],cal)
print('inverse ranges',z.min(0),z.max(0))
def op(u,v,mode):
 def tx(z,p):
  k=p[0]; f=k*z
  if mode=='cos': return np.cos(f)
  if mode=='soft': return z/(1+abs(f))
  if mode=='softlin': return z/(1+abs(f))+p[1]*z
  if mode=='tanhlin':return np.tanh(f)+p[1]*z
  if mode=='asin':return np.arcsinh(f)
  if mode=='sat':return z/(1+f*f)**.5
  if mode=='sq':return np.c_[z,z*z]
  if mode=='quad':return quad(z)[:,1:]
 def fit(p):return ls(af(tx(u,p)),v)
 def res(p):return (af(tx(u,p))@fit(p)-v).ravel()
 p=min([least_squares(res,[k,0.] if mode.endswith('lin') else [k],max_nfev=300) for k in (.5,1.4,2.6)],key=lambda o:o.cost).x
 return p,rm(res(p)),lambda z:af(tx(z,p))@fit(p)
b={};
for panel,key,modes in [('panel_b','v',['cos']),('panel_c','v1',['soft','softlin','tanhlin','asin','sat']),('panel_c','v2',['sq','quad'])]:
 for mode in modes:
  p,e,f=op(s[panel]['u'],s[panel][key],mode); print(panel,key,mode,p,e); b[mode]=f
Y,T=[s['train'][k] for k in ('y','traj')]
def check(name,h):
 a=af(h); d=quad(h) if h.shape[1]<=8 else a
 def score(a,y):
  e=a[52:]@ls(a[:52],y[:52])-y[52:];return np.mean(abs(e)),rm(a[:52]@ls(a[:52],y[:52])-y[:52])
 print(name,'T',np.round(score(a,T),5),'Y',np.round(score(a,Y),5),'YQ',np.round(score(d,Y),5))
check('raw',z)
for name,f in b.items():
 for parent,w in [('z',z),('B',b['cos'](z)),('Csoft',b['soft'](z))]:
  h=f(w); check(name+'('+parent+')',h); check('z+'+name+'('+parent+')',np.c_[z,h]);
  if name!='cos':check('B+'+name+'('+parent+')',np.c_[b['cos'](z),h])
