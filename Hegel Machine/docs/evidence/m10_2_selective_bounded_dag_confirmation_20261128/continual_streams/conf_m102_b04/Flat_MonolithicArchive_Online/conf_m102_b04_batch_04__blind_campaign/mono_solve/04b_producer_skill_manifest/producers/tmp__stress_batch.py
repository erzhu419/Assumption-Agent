import importlib.util, pickle, numpy as np
p='/holdout_task/batch_solver.py'; spec=importlib.util.spec_from_file_location('batch_solver',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; z=tr['traj']; n=len(x); idx=np.arange(n)
u=m._coordinates(s['panel_a'],x)
rng=np.random.default_rng(621); folds=[('52/20',idx[:52],idx[52:])]
folds += [(f'random{k}',q[20:],q[:20]) for k in range(8) for q in [rng.permutation(n)]]
folds += [(f'axis{k}{side}',np.setdiff1d(idx,j),j) for k in range(4) for side,j in [('low',u[:,k].argsort()[:10]),('high',u[:,k].argsort()[-10:])]]
folds += [('largest-radius',np.setdiff1d(idx,j),j) for j in [np.linalg.norm(u,axis=1).argsort()[-12:]]]
for mode in ('exponential','svd','raw'):
    errors=[]
    for name,i,j in folds:
        st=dict(s,train={'x':x[i],'y':y[i],'traj':z[i]},test={'x':x[j]}); a,b=m._predict(st,mode)
        e=abs(b-z[j]); errors.append(np.r_[abs(a-y[j]).mean(),e.mean(),e[:,-1].mean(),e.mean(axis=0).max()]);
        print(mode,name,np.round(errors[-1],6))
    print(mode,'worst all',np.max(errors,axis=0),'worst random',np.max(errors[:9],axis=0))
result=m.solve_batch(s); assert all(np.isfinite(v).all() for v in result.values()); print('full-output diagnostics',result['diagnostics_vec'])
