import numpy as np,pickle,runpy
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; w=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s})['w'][:72]; y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; x=s['train']['x']; c=s['panel_c']; u=c['u']; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]; aff=lambda a:np.column_stack([np.ones(len(a)),a]); print('t',t,'w range',w.min(0),w.max(0),'y range',y.min(),y.max()); print('traj SVD',np.linalg.svd(T,compute_uv=False)); print('panel C range',[(k,v.min(0),v.max(0)) for k,v in c.items()]); print('panel correlations',np.round(np.corrcoef(np.c_[u,c['v1'],c['v2']].T)[:4,4:],2)); print('panel fit',[(k,np.std(v-aff(u)@ls(aff(u),v),axis=0)) for k,v in c.items() if k!='u']); print('traj PCA vectors',np.round(np.linalg.svd(T)[2][:5],3)); print('traj diff svd',np.linalg.svd(np.diff(np.c_[np.zeros(len(T)),T]),compute_uv=False));
def score(F,targets=(y,T)):
 return [(round(np.mean(abs((F[:52]@ls(F[:52],Y[:52])-Y[:52]))),5),round(np.mean(abs(F[52:]@ls(F[:52],Y[:52])-Y[52:])),5)) for Y in targets]
for name,z in [('w',w),('x',x)]:
 for d in range(1,4):print(name,d,score(PolynomialFeatures(d).fit_transform(z)))
for name,f in [('atan',np.arctan),('asinh',np.arcsinh),('inv3',lambda u:np.sinh(np.arcsinh(u)/3)),('tanh',np.tanh),('sin',np.sin),('softsign',lambda u:u/(1+abs(u))),('cube',np.cbrt)]:
 for g in [1,2,4,8]:
  z=f(g*w);print(name,g,score(np.c_[PolynomialFeatures(2).fit_transform(z),w]))
for f in [lambda a:a,lambda a:np.arctanh(np.clip(a,-.999999,.999999)),lambda a:np.arcsin(np.clip(a,-1,1))]:
 z=f(y);print('y link',z.min(),z.max(),score(PolynomialFeatures(2).fit_transform(w),(z,)))
