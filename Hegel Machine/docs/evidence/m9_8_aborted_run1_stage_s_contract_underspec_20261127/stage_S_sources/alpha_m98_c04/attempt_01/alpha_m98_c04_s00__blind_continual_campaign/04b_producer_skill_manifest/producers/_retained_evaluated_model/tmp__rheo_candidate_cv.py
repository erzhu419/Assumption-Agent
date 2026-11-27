exec(open('/tmp/rheo_probe.py').read().split('for kind in')[0])
lab, st = pi['lab_id'], pi['stratum']
cell = np.unique(np.column_stack((lab, st)),axis=0,return_inverse=True)[1]
for name,tags in [('coupon',np.arange(len(lab))),('cell',cell),('lab',lab)]:
    pred=np.empty_like(s._obs(pi))
    for k in np.unique(tags):
        ix=tags==k
        fit={key:v[~ix] for key,v in pi.items()}
        val={key:v[ix] for key,v in pi.items()}
        d=dict(p, test=val, chamber_verification_trials=fit)
        o=s.solve_batch(d)
        pred[ix]=np.column_stack((o['predicted_strain']-val['strain_0'][:,None],o['initial_rate_pred']))
    r=abs(pred-s._obs(pi)); c,v=r[:,:-1],r[:,-1]
    print(name,'p90',np.round([np.quantile(c,.9),np.quantile(v,.9)],4),'worst cell MAE',np.round([max(c[cell==k].mean() for k in np.unique(cell)),max(v[cell==k].mean() for k in np.unique(cell))],4))
o=s.solve_batch(dict(p,chamber_verification_trials=None))
assert all(np.all(np.isfinite(v)) for v in o.values())
assert .05 < o['estimated_taus'][0] < o['estimated_taus'][1] < 25
print('blind-path output shapes', {k:v.shape for k,v in o.items()})
