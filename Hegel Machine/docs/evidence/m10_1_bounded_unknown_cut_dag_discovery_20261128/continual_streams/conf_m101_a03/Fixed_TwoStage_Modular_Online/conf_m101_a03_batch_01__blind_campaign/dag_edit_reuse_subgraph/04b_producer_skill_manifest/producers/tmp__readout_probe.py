import pickle,itertools as it,numpy as np
from scipy.optimize import least_squares
p='/holdout_task/_frozen_nodes/node_00.py'; ns={};exec(compile(open(p).read(),p,'exec'),ns)
d=ns['run_node']({'__input__':pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']});z=d['z'];y=d['y'];n=len(z);a=np.c_[np.ones(n),z,z[:,0]*z[:,2],z[:,1]*z[:,3]]
def tanh(x):return np.tanh(x)
def sin(x):return np.sin(x)
def rat(x):return x/(1+x*x)
def fit(fs, start, ids=np.arange(52), extra=None):
 def design(p):
  cols=[f(z@v[:4]+v[4]) for f,v in zip(fs,p.reshape(-1,5))]
  return np.column_stack([a]+cols+([] if extra is None else [extra]))
 def coef(x):return np.linalg.lstsq(x[ids],y[ids],rcond=None)[0]
 def res(p):
  x=design(p);return x[ids]@coef(x)-y[ids]
 r=least_squares(res,start,max_nfev=500);x=design(r.x);w=coef(x);e=y-x@w
 return (np.mean(abs(e[ids])),np.mean(abs(e[np.setdiff1d(np.arange(n),ids)])),r.x,w,e)
for f in (tanh,sin,rat):
 r=fit([f],[1,1,1,-1,0]);print('single',f.__name__,r[:4]);
for f,g in it.product((tanh,sin,rat),repeat=2):
 r=fit([f,g],[1,1,1,-1,0,2,2,2,0,0]);print('double',f.__name__,g.__name__,r[:4]);
