import importlib.util,pickle,numpy as np
sp=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; ids=np.arange(len(tr['y']))
folds=[ids[52:],ids[56:]]+[b for seed in (111,927) for b in np.array_split(np.random.RandomState(seed).permutation(ids),4)]
results=[]
for b in folds:
    a=np.setdiff1d(ids,b)
    out=m.solve_batch(dict(s, train={**tr,**{k:tr[k][a] for k in ('x','y','traj')}}, test={'x':tr['x'][b]}))
    y=out['y_hat']; state=out['state_pred']; de=abs(state-tr['traj'][b]); last=state[:, -1]
    np.testing.assert_allclose(out['diagnostics_vec'],[y.mean(),y.std(),last.mean(),last.std()],rtol=0,atol=1e-12)
    results.append([abs(y-tr['y'][b]).mean(),de.mean(),de.mean(0).max()])
    print(len(b),np.round(results[-1],6))
print('Worst split y / trajectory / timestamp:',np.max(results,axis=0))
assert np.max(np.array(results)[:,:2]) <= .038
p=s['panel_a']; u,v=p['u'],p['v']
def forward(u,c):
    w=np.roll(u,-1,axis=1)
    return c[0]+c[1]*u+c[2]*w+c[3]*u*w+c[4]*u**3
r=np.linalg.norm(forward(u,m._calibrate(p))-v,axis=1); clean=np.flatnonzero(r<.01)
errs=[]
for i in clean:
    a=np.arange(len(u))!=i; q=np.setdiff1d(clean,[i]); d=np.c_[np.ones(len(u)),u]
    aff=d[i]@np.linalg.lstsq(d[q],v[q],rcond=None)[0]
    pred=forward(u[i:i+1],m._calibrate({'u':u[a],'v':v[a]}))[0]
    errs.append([abs(aff-v[i]).mean(),abs(pred-v[i]).mean()])
print('Panel A clean rows:',len(clean),'LOO MAE affine / cyclic cubic:',np.mean(errs,axis=0),'worst cubic row MAE:',np.max(errs,axis=0)[1])
assert len(clean)==45 and np.max(errs,axis=0)[1]<.001
