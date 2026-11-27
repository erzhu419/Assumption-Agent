exec(open('/tmp/m98_final_probe.py').read().split('A,base=data()')[0])
train = d['train']
tr, te = m._prepare(d)
dt = d['past_dt']
yc = train['strain_curves_obs'] - train['strain_0'][:,None]
yr = train['instant_rate_obs']
lt = train['lab_id']; st = train['stratum']
mc = np.isfinite(train['coupon_features_raw']).all(1)
folds = [(f'lab {v}',lt==v) for v in np.unique(lt)] + [(f'stratum {v}',st==v) for v in np.unique(st)] + [('missing',~mc)]
rows=[]
for name, mask in folds:
    keep = ~mask
    ti, wi = m._fit(dt,t,tuple(v[keep] for v in tr),yc[keep],yr[keep])
    C,R=m._design(ti,dt,t,tuple(v[mask] for v in tr))
    ec=np.abs(C@wi-yc[mask]); er=np.abs(R@wi-yr[mask])
    row=[np.max(np.abs(ti/ts-1)),np.median(ec),np.median(er)]
    assert .05 < ti[0] < ti[1] < 25 and np.isfinite(row).all()
    rows.append(row)
    print(name, 'max tau relative shift, median strain and rate errors',np.round(row,6))
print('intrinsic worst blocked',np.max(rows,axis=0))
# A mixing/offset change to the raw bridge must not change any physical output when the calibration sweeps change with it.
from copy import deepcopy
shifted=deepcopy(d)
M=np.array([[1.2,.1,0,0],[0,.8,.09,0],[.04,0,1.15,0],[0,.05,0,.95]])
a=np.array([.11,-.04,.07,.02])
bridge=shifted['calibration_load_cells']; bridge['bridge_raw']=bridge['bridge_raw']@M+a
for name in ('train','test'):
    for key in ('past_channel_raw','hold_channel_raw'):
        shifted[name][key]=shifted[name][key]@M+a
for key, value in m.solve_batch(shifted).items():
    np.testing.assert_allclose(value,out[key],rtol=5e-6,atol=5e-6,err_msg=key)
# Compare deployment rates with the derivative of the actual strain output.
zz=dict(d); zz['eval_times']=np.array([0.,1e-6]); zz['train']=dict(train)
# Keep the training time grid unchanged; evaluate the exact chamber design at both times.
hold,features=te[1:]; C,R=m._chamber_design(ts,zz['eval_times'],hold,features@w[2:]); fast,slow=.34880024,1.63895287
coef=np.array([w[0],fast,w[1],slow,1-fast/ts[0]-slow/ts[1]])
np.testing.assert_allclose(((C@coef)[:,1]-(C@coef)[:,0])/1e-6,out['initial_rate_pred'],rtol=2e-6,atol=2e-6)
print('Bridge mixing invariance and zero-time derivative passed')
