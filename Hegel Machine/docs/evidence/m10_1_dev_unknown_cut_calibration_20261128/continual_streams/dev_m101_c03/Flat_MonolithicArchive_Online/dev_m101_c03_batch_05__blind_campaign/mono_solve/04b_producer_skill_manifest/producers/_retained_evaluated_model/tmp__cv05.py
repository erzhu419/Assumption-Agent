import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; p=s['panel_c']; u=p['u'];v=p['v2']; f=m._second_operator(u,v);print('C2 params',f.__closure__ and dict(zip(f.__code__.co_freevars,[c.cell_contents for c in f.__closure__]))['p']);print('C1 and C2 panel RMSE',np.mean((m._cross(u)@m._fit(m._cross(u),p['v1'])-p['v1'])**2)**.5,np.mean((f(u)-v)**2)**.5,'C2 LOO',np.mean([(m._second_operator(np.delete(u,i,0),np.delete(v,i,0))(u[i:i+1])-v[i])**2 for i in range(len(u))])**.5)
z=m._decode(s['panel_a'],tr['x']); radius=np.linalg.norm(z,axis=1); n=len(z)
for name,h in [('56/16',np.arange(56,n)), *[(f'random {i}',h) for i,h in enumerate(np.array_split(np.random.default_rng(25).permutation(n),5))],('outer radius',np.argsort(radius)[-16:]),*[(f'extreme {j} {sign}',np.argsort(sign*z[:,j])[-12:]) for j in range(4) for sign in [-1,1]]]:
 ix=np.setdiff1d(np.arange(n),h); p=m.solve_batch(dict(s,train=dict(tr,**{k:tr[k][ix] for k in ('x','y','traj')}),test={'x':tr['x'][h]})); dy=abs(p['y_hat']-tr['y'][h]); d=abs(p['state_pred']-tr['traj'][h]); print(name, 'y/traj/end/worst time',np.round([dy.mean(),d.mean(),d[:,-1].mean(),d.mean(0).max()],7))
