import pickle, runpy, numpy as np
path = '/holdout_task/'
s = pickle.load(open(path+'public_batch.pkl', 'rb'))[0]['study']
a = runpy.run_path(path+'batch_solver.py'); solve = a['solve_batch']
tr=s['train']; x=tr['x']; ids=np.arange(len(x)); times=s['eval_times'];
cal=a['_calibration'](s['panel_a']); z=a['_decode'](x,cal); aff=a['_affine']; b=s['panel_b']; w=aff(z)@a['_fit'](aff(b['u']),b['v']);
seq=np.random.default_rng(11).permutation(len(x))
splits=[('public52/20',ids[:52],ids[52:]),('reverse52/20',ids[20:],ids[:20])]
for k,j in enumerate(np.array_split(seq,4)):
    splits.append(('random'+str(k),np.setdiff1d(ids,j),j))
for label, value in [('input-extreme',np.linalg.norm(w,axis=1)),('paired-extreme',abs(a['_readout'](w)[:,-1]))]+[(f'channel{k}{side:+d}',side*w[:,k]) for k in range(4) for side in (-1,1)]:
    i=np.argsort(value); splits.append((label,i[:-12],i[-12:]))

def check(out,n):
    y,t,d=[out[k] for k in ('y_hat','state_pred','diagnostics_vec')];
    assert all(isinstance(v,np.ndarray) and np.isfinite(v).all() for v in (y,t,d));
    assert y.shape==(n,) and t.shape==(n,len(times)) and d.shape==(4,)
    np.testing.assert_allclose(d,[y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std()],atol=1e-13,rtol=1e-13)
    return y,t

worst=[]
for label,i,j in splits:
    train={k: v[i] for k,v in tr.items() if k!='summary'}
    y, t=train['y'],train['traj']; train['summary']=np.array([y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std()]);
    yh, th=check(solve(dict(s,train=train,test={'x':x[j]})),len(j));
    em=abs(yh-tr['y'][j]); tm=abs(th-tr['traj'][j]); worst.append((em.mean(),tm.mean(),tm.mean(0).max()))
    print(label, 'y MAE', round(em.mean(),6), 'max y', round(em.max(),6), 'traj MAE',round(tm.mean(),6),'worst horizon',round(tm.mean(0).max(),6))
assert max(row[0] for row in worst)<=.038
full=check(solve(s),len(s['test']['x'])); check(solve(dict(s,test={'x':s['test']['x'][:1]})),1)
print('worst slice metrics: y, state, horizon =',np.max(worst,axis=0)); print('test y range:',full[0].min(),full[0].max())
