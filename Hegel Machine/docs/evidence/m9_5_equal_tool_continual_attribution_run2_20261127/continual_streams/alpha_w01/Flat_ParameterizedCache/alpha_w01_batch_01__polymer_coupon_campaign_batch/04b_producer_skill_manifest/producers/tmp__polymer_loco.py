exec(open('/tmp/polymer_eval.py').read().split("print('taus predicted")[0])
tr=d['train']; lab=tr['lab_id']; keep=np.isfinite(tr['coupon_features_raw']).all(axis=1)
for k in range(4):
    subset=lambda drop: {key: val[drop] for key,val in tr.items()}
    fit=m.solve_alpha_w01_batch_01(dict(d,train=subset(lab!=k)))
    j=q['lab_id']==k
    error=lambda key,target: np.sqrt(np.mean((fit[key][j]-g[target][j])**2))
    print('held-out training lab',k,'taus',fit['estimated_taus'],'strain',error('predicted_strain','te_clean'),'secant',error('secant_compliance','te_secant'),'hazard',error('hazard_crossing_times','te_cross'))
