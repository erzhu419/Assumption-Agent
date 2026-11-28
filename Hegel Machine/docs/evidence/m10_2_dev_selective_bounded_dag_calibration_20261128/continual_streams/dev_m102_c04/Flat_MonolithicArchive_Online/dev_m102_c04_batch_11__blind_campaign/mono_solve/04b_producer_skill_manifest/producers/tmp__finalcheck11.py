exec(open('/tmp/check11.py').read().split("print('cross coeff'")[0])
splits=[('required',np.arange(52,72))]
splits += [(f'fold{i}',v) for i,v in enumerate(np.array_split(np.random.default_rng(11).permutation(72),4))]
splits += [(f'tail{i}',np.argsort(abs(z[:,i]))[-20:]) for i in range(4)]
splits += [('high-radius',np.argsort(np.linalg.norm(z,axis=1))[-20:])]
metrics=[]
for label,te in splits:
    ix=np.setdiff1d(np.arange(len(x)),te)
    train={k:tr[k][ix] for k in ('x','y','traj')}
    train['summary']=np.array([y[ix].mean(),y[ix].std(),T[ix,-1].mean(),T[ix,-1].std()])
    out=m.solve_batch(dict(s,train=train,test={'x':x[te]}))
    pred, state, diag=map(out.get,('y_hat','state_pred','diagnostics_vec'))
    assert pred.shape==(len(te),) and state.shape==(len(te),T.shape[1])
    assert all(np.isfinite(v).all() for v in (pred,state,diag))
    np.testing.assert_array_equal(diag,[pred.mean(),pred.std(),state[:,-1].mean(),state[:,-1].std()])
    e=abs(state-T[te]); vals=(abs(pred-y[te]).mean(),e.mean(),e.mean(0).max(),e[:,-1].mean())
    print(label,*[f'{v:.6f}' for v in vals]); metrics.append(vals)
worst=np.max(metrics,axis=0); print('worst',*[f'{v:.6f}' for v in worst]); assert np.all(worst <= .038)
out=m.solve_batch(s)
assert out['y_hat'].shape==(36,) and out['state_pred'].shape==(36,8) and out['diagnostics_vec'].shape==(4,)
assert all(np.isfinite(v).all() for v in out.values())
