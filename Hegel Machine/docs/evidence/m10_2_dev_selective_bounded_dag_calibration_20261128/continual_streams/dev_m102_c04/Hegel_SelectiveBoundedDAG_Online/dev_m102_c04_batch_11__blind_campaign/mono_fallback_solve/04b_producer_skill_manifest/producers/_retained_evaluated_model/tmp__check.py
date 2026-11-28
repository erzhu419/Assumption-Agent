import numpy as np,pickle,importlib.util
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];T=s['train'];i=np.arange(len(T['x']));x=T['x']; X=m._coordinates(s['panel_a'],x); p,r=m._operators(s['panel_c'],X); J=(p[:,:,None]*r[:,None,:]); a=m._affine(np.c_[p,r,J.reshape(len(x),16)]); w=m._fit(a,T['traj']); print('interaction coefficients [first,last]');print(np.round(w[9:,[0,-1]].reshape(4,4,2),3));print('linear coefficients [first,last]');print(np.round(w[:9,[0,-1]],3)); print('y weights',m._fit(m._affine(np.c_[p,r,np.trace(J,axis1=1,axis2=2)]),T['y']))
def cv(tr,te):
 S=dict(s,train={**T,**{k:T[k][tr] for k in ('x','y','traj')}},test={'x':x[te]});o=m.solve_batch(S)
 return np.mean(abs(o['y_hat']-T['y'][te])),np.mean(abs(o['state_pred']-T['traj'][te])),np.max(np.mean(abs(o['state_pred']-T['traj'][te]),axis=0))
print('ordered',cv(i[:52],i[52:])); R=np.random.default_rng(1729); random=[]
for _ in range(20):
 j=R.permutation(i);random.append(cv(j[:52],j[52:]))
print('random mean',np.mean(random,0),'worst',np.max(random,0));print('edge slices')
for v in [*x.T,np.linalg.norm(x,axis=1),T['y'],T['traj'][:,-1]]:
 j=np.argsort(v);print(cv(j[:52],j[52:]),cv(j[20:],j[:20]))
