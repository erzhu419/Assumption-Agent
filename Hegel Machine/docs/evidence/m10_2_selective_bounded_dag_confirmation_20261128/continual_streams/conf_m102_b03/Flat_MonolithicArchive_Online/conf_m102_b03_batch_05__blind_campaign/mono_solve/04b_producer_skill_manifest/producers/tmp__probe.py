import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
one=lambda x:np.column_stack([np.ones(len(x)),x])
u,v=(s['panel_a'][k] for k in ['u','v']); a=one(u); good=np.ones(len(u),bool)
for _ in range(6):
 coef=ls(a[good],v[good]); r=np.linalg.norm(a@coef-v,axis=1); good=r<4*np.median(r)
print('align',coef,'residuals',np.quantile(r,[0,.25,.5,.75,.9,.93,.95,1])); print('times,telemetry',s['eval_times'],s['telemetry_vec'])
x=s['train']['x'];z=np.linalg.solve(coef[1:].T,(x-coef[0]).T).T
b=ls(one(s['panel_b']['u']),s['panel_b']['v']); w=one(z)@b
Y=np.column_stack([s['train']['y'],s['train']['traj']]); t=s['eval_times']; T=Y[:,1:];
for fam in [lambda p:-np.expm1(-t[:,None]*p),lambda p:np.exp(-t[:,None]*p),lambda p:np.c_[t,-np.expm1(-t*p[0])]]:
 def fun(p):
  h=fam(p);return (h@ls(h,T.T)-T.T).ravel()
 f=least_squares(fun,[1.8,.4],bounds=(.001,20));print('time fit',f.x,np.mean(abs(f.fun)))
for name,d in [('raw',x),('align',z),('b',w)]:
 D=lambda deg:np.column_stack([np.ones(len(d))]+[np.prod(d[:,j],axis=1) for k in range(1,deg+1) for j in combinations_with_replacement(range(d.shape[1]),k)])
 for deg in [1,2,3]:
  q=D(deg); e=abs(q[52:]@ls(q[:52],Y[:52])-Y[52:]); print(name,deg,e[:,0].mean(),e[:,1:].mean())
