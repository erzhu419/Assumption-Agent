exec(open('/tmp/stress05.py').read().split('m=load(1);')[0])
m=load(1)
def pred(i,j):
    out=m.run_node({'__input__':{'train':{'y':y[i],'traj':z[i]}},'node_00':{'cubic_train':f[i],'cubic_test':f[j]}})
    yh, zh = out['y_hat'], out['state_pred']
    np.testing.assert_allclose(out['diagnostics_vec'], [yh.mean(), yh.std(), zh[:,-1].mean(), zh[:,-1].std()])
    return np.column_stack((yh, zh))
r=score(pred)
print('Exact public, random maxima, tail maxima, worst random horizon:',r)
print('Public per-horizon MAE:',abs(pred(*splits[0])[:,1:]-z[splits[0][1]]).mean(0))
assert np.isfinite(r).all() and max(r[:2])<=.038 and max(r[2:4])<.038 and max(r[4:6])<.08 and r[6]<.04943502
