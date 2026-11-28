import pickle, numpy as np, importlib.util,glob
from itertools import combinations_with_replacement as cwr
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,z=[s['train'][k] for k in ('x','y','traj')]
np.set_printoptions(suppress=True,precision=4,linewidth=160)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def poly(a,d=2):return np.column_stack([np.prod(a[:,k],axis=1) for j in range(d+1) for k in cwr(range(a.shape[1]),j)])
def err(r):return np.mean(abs(r),axis=0)
def load(f):
 spec=importlib.util.spec_from_file_location('old',f);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 m=load(f); sub=dict(s,train={**s['train'],**{k:s['train'][k][:52] for k in ('x','y','traj')}},test={'x':x[52:]});out=m.solve_batch(sub);print(f[-18:],err(out['y_hat']-y[52:]),err(out['state_pred']-z[52:]).mean());
 if '06_solver' in f: m6=m
u,v=[s['panel_a'][k] for k in ('u','v')];co=m6._fit_calibration(s['panel_a']);print('a co',co[np.any(co,axis=1)]);print('a res',np.sort(np.linalg.norm(v-poly(u,3)@co,axis=1)));q=m6._invert(x,co)
features={'x':x,'q':q,'b_old':m6._panel_b(s['panel_b'],q)}
for name in ('panel_b','panel_c'):
 d=s[name]; u=d['u'];
 for key,v in d.items():
  if key=='u':continue
  for tag,fun in [('aff',lambda a:poly(a,1)),('quad',poly),('cyclic',lambda a:np.c_[poly(a,1),a*a,a*np.roll(a,-1,axis=1)]),('sep',lambda a:np.c_[poly(a,1),a*a,a**3]),('odd',lambda a:np.c_[poly(a,1),a**3,np.tanh(a)]),('all3',lambda a:poly(a,3))]:
   f=fun(u);w=ls(f,v);h=(f*np.linalg.pinv(f).T).sum(1);print(name,key,tag,'fit',err(f@w-v).mean(),'LOO',err((f@w-v)/np.maximum(1-h[:,None],1e-9)).mean());
   if tag!='all3':features[name+key+tag]=fun(q)@w
for tag,a in features.items():
 print(tag,[(round(err(f[52:]@ls(f[:52],y[:52])-y[52:]),5),round(err(f[52:]@ls(f[:52],z[:52])-z[52:]).mean(),5)) for f in [poly(a,1),poly(a),np.c_[poly(a,1),a*np.roll(q,-1,axis=1)]]]);
for a in [x,q]:
 f=poly(a);print('y quad coef',ls(f,y));print('z last quad coef',ls(f,z[:,-1]));
