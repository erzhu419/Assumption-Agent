import pickle, runpy, numpy as np
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; f=runpy.run_path('/holdout_task/node_01.py')['run_node']; enc=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']; x=st['train']['x']; y=st['train']['y']; traj=st['train']['traj']; n=len(y); ids=np.arange(n); z=enc({'__input__':st})['z']; rng=np.random.default_rng(24)
def run(j):
 i=np.setdiff1d(ids,j); batch=dict(st,train={'x':x[i],'y':y[i],'traj':traj[i]},test={'x':x[j]}); out=f({'__input__':batch,'node_00':enc({'__input__':batch})}); ey=abs(out['y_hat']-y[j]); et=abs(out['state_pred']-traj[j]); return ey.mean(),et.mean(),ey.max(),et.mean(0).max()
for kind, groups in [('random', [g for _ in range(5) for g in np.array_split(rng.permutation(n),6)]),('feature-tail', [ix for k in range(z.shape[1]) for ix in (z[:,k].argsort()[:12],z[:,k].argsort()[-12:])]),('contiguous',np.array_split(ids,6))]:
 v=np.array([run(j) for j in groups]); print(kind,'mean',v.mean(0),'worst',v.max(0))
