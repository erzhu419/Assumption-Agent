import importlib.util,pickle,numpy as np
sp=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train']; x,y,T=[tr[k] for k in ['x','y','traj']]; n=len(y)
rng=np.random.default_rng(21); allidx=np.arange(n); z=m._coords(s['panel_a'],x); F=m._design(*m._operators(s['panel_c'],z));
splits=[('public',allidx[52:])]+[(f'random{i}',rng.choice(n,20,replace=False)) for i in range(24)]+[(f'extreme{j}{h}',np.argsort(z[:,j])[::h][:20]) for j in range(4) for h in [-1,1]]
out=[]
for label,b in splits:
 a=np.setdiff1d(allidx,b); r=m.solve_batch(dict(s,train={k:tr[k][a] for k in ['x','y','traj']},test={'x':x[b]})); e=np.abs(r['y_hat']-y[b]); v=np.abs(r['state_pred']-T[b]);out.append((label,e.mean(),v.mean(),v.mean(0).max(),e.max(),v.max()))
for group in [out[:1],out[1:25],out[25:]]:print('slices',[(group[np.argmax([g[j] for g in group])][0],max(g[j] for g in group)) for j in range(1,6)])
print('readout coefficients (rounded)',np.round(m._ls(F,np.c_[y,m._ls(m._time(s['eval_times'],T),T.T).T]),4))
