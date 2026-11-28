import pickle,runpy,numpy as np
from itertools import combinations_with_replacement
from scipy.linalg import svd
from scipy.optimize import minimize_scalar
np.set_printoptions(precision=4,suppress=True,linewidth=140)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
m=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py'); z=m['run_node']({'__input__':s})['z_tr']; a,b,c=[s['panel_'+k] for k in 'abc']; u=c['u']; v=np.column_stack((c['v1'],c['v2'])); y=s['train']['y']; t=s['eval_times']; tr=s['train']['traj']; x=s['train']['x']
def design(x,n):
 return np.column_stack([np.prod(x[:,i],axis=1) for d in range(n+1) for i in combinations_with_replacement(range(x.shape[1]),d)])
def ls(d,v): return np.linalg.lstsq(d,v,rcond=None)[0]
def press(d,v):
 q=np.linalg.svd(d,full_matrices=False)[0]; r=v-q@(q.T@v); return np.mean(np.abs(r/(1-(q*q).sum(1)[:,None])),axis=0)
print('telemetry',s['telemetry_vec'],'times',t)
print('ranges',np.min(z,0),np.max(z,0),'y',y.min(),y.max(),'traj',tr.min(0),tr.max(0)); print('trajectory SVD',svd(tr)[1]); print('mean traj',tr.mean(0)); print('cross corr y-tr',np.corrcoef(np.column_stack((y,tr)).T)[0])
for j,(name,vv) in enumerate([('C1',v[:,:4]),('C2',v[:,4:]),('C1->C2',v[:,4:])]):
 uu=u if j<2 else v[:,:4]; dd=design(uu,1); print(name,'affine coeff',ls(dd,vv).T,'LOO',press(dd,vv),'quad',press(design(uu,2),vv));
 for nm,f in [('cubic',lambda x,k:x**3),('gauss',lambda x,k:x*(-np.expm1(-k*x*x))),('tanh',lambda x,k:np.tanh(k*x)),('sin',lambda x,k:np.sin(k*x)),('rat',lambda x,k:x/(1+k*x*x)),('square',lambda x,k:x*x)]:
  fn=lambda k: np.column_stack((dd,f(uu,k)))
  o=minimize_scalar(lambda k: np.mean(press(fn(k),vv)),bounds=(.08,3),method='bounded'); print(nm,round(o.x,4),press(fn(o.x),vv))
coef,seed=m['_fit_a'](a['u'],a['v']); ref=m['_invert_a'](x,coef,seed); k,g=m['_fit_b'](b['u'],b['v']); print('B k/gain',k,g.T)
for nm,xx in [('raw',x),('ref',ref),('z',z)]:
 for d in range(1,5):
  X=design(xx,d); Y=np.column_stack((y,tr)); e=np.abs(X[52:]@ls(X[:52],Y[:52])-Y[52:]); print(nm,d,'split',e[:,0].mean(),e[:,1:].mean(),'LOO',press(X,Y) if X.shape[1]<len(X) else '-')
print('trajectory first 12 with y and z'); print(np.column_stack((z,y,tr))[:12]);
