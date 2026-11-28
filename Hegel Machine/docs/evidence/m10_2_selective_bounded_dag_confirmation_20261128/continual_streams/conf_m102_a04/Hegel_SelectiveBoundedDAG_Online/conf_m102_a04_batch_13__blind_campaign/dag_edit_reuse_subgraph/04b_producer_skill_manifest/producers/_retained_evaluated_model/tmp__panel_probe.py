import pickle, numpy as np, importlib.util
from scipy.optimize import minimize_scalar, least_squares
p='/tmp/hegel_continual_archive/conf_m102_a04_batch_00_solver.py'
s=importlib.util.spec_from_file_location('base',p); b=importlib.util.module_from_spec(s); s.loader.exec_module(b)
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=st['panel_c']['u']; v=st['panel_c']['v2']; n=len(u); rng=np.random.default_rng(521); folds=np.array_split(rng.permutation(n),6)
lin=b.lin
bases={
 'affine':(lambda x,q:b.one(x),None),
 'quad':(lambda x,q:b.one(np.c_[x,x*x]),None),
 'quartic':(lambda x,q:b.one(np.c_[x,x*x,x**4]),None),
 'rational_archive':(lambda x,q:b.one(np.c_[x,x*x,x*x/(1+q*x*x)]),(.08,.65)),
 'rational_wide':(lambda x,q:b.one(np.c_[x,x*x,x*x/(1+q*x*x)]),(.005,15)),
 'cos':(lambda x,q:b.one(np.cos(q*x)),(.05,9)),
 'cos+linear':(lambda x,q:b.one(np.c_[x,np.cos(q*x)]),(.05,9)),
 'cos+quad':(lambda x,q:b.one(np.c_[x,x*x,np.cos(q*x)]),(.05,9)),
 'sin':(lambda x,q:b.one(np.sin(q*x)),(.05,9)),
 'sin+linear':(lambda x,q:b.one(np.c_[x,np.sin(q*x)]),(.05,9)),
 'fourier':(lambda x,q:b.one(np.c_[np.cos(q*x),np.sin(q*x)]),(.05,9)),
 'gaussian':(lambda x,q:b.one(np.exp(-q*x*x)),(.005,15)),
 'gaussian+linear':(lambda x,q:b.one(np.c_[x,np.exp(-q*x*x)]),(.005,15)),
 'cosh+linear':(lambda x,q:b.one(np.c_[x,np.cosh(q*x)]),(.05,6)),
 'tanh+linear':(lambda x,q:b.one(np.c_[x,np.tanh(q*x)]),(.05,9)),
 'atan+linear':(lambda x,q:b.one(np.c_[x,np.arctan(q*x)]),(.05,9)),
 'exp+linear':(lambda x,q:b.one(np.c_[x,np.exp(q*x)]),(-6,6)),
 'pairwise':(lambda x,q:b.one(np.c_[x,*(x[:,i]*x[:,j] for i in range(4) for j in range(i,4))]),None),
}
def fit(fun,box,I):
 def eval(q):
  f=fun(u[I],q); c=lin(f,v[I]); return np.mean((f@c-v[I])**2),c
 if box is None:q=0
 else:
  grid=np.linspace(*box,120); k=np.argmin([eval(q)[0] for q in grid]); q=minimize_scalar(lambda q:eval(q)[0],bounds=(grid[max(0,k-1)],grid[min(k+1,len(grid)-1)]),method='bounded',options={'xatol':1e-12}).x
 return q,eval(q)[1]
for name,(fun,box) in bases.items():
 q,c=fit(fun,box,np.arange(n)); errors=[]
 for J in folds:
  qi,ci=fit(fun,box,np.setdiff1d(np.arange(n),J)); errors.append(np.abs(fun(u[J],qi)@ci-v[J]))
 print(name,'q',round(q,8),'fit',round(np.mean(np.abs(fun(u,q)@c-v)),8),'CV',round(np.mean(errors),8),'worst fold',round(max(np.mean(e) for e in errors),8))
a,sc=b._align(st['panel_a']); r=np.abs(st['panel_a']['v']-a-st['panel_a']['u']*sc); print('align offset',a,'scale',sc,'resid percentiles',np.percentile(r,[25,50,75,90,100]));
train=st['train']; t=st['eval_times']; tr=train['traj']; basis=b._time_basis(t,tr); print('trajectory basis projection MAE',np.mean(abs((basis@lin(basis,tr.T)).T-tr)))
