import numpy as np,pickle,runpy,copy
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; f=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']; m=runpy.run_path('/holdout_task/node_01.py');o=f({'__input__':s});w=m['_operator'](s,next(v for v in o.values() if np.shape(v)==(108,4)))[:72]; ls=m['_ls'];aff=m['_aff'];
funcs={'lin':lambda x:x,'asinh':np.arcsinh,'atan':np.arctan,'tanh':np.tanh,'inv3':lambda x:np.sinh(np.arcsinh(x)/3),'sin':np.sin,'erf':__import__('scipy.special',fromlist=['erf']).erf,'softsign':lambda x:x/np.sqrt(1+x*x)}
for label in ('y','traj'):
 y=s['train'][label];sc=np.maximum(y.std(axis=0),1e-8);print(label)
 for k,f in funcs.items():
  for mode in (0,1,2,3):
   def basis(g):
    z=f(g*w);a=aff(z) if mode in (0,2) else aff(np.c_[w,z]);return np.c_[a,*[z[:,i]*z[:,j] for i in range(4) for j in range(i,4)]] if mode>=2 else a
   scores=[]
   for val in [np.arange(i,i+18) for i in (0,18,36,54)]+[np.arange(52,72)]:
    tr=np.setdiff1d(np.arange(72),val)
    def fit(g,predict=False):
     a=basis(g);b=ls(a[tr],y[tr]);return np.mean(abs(a[val]@b-y[val])) if predict else np.mean(((a[tr]@b-y[tr])/sc)**2)
    g=minimize_scalar(fit,bounds=(0.2,9),method='bounded').x;scores.append((fit(g,True),g))
   print(k,mode,np.round(np.array(scores).T,4).tolist())
