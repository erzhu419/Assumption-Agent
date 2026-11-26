import importlib.util, pickle, numpy as np

def load(path):
    spec = importlib.util.spec_from_file_location('solver', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.solve_alpha_w01_batch_08

rm = lambda a: float(np.sqrt(np.mean(a*a)))
for o in pickle.load(open('/holdout_task/public_alpha_w01_batch_08.pkl', 'rb')):
    d,g=o['dataset'],o['gt']; te=d['test']; labs=te['lab_id']
    missing=~np.isfinite(te['coupon_features_raw']).all(axis=1)
    hist,hold = g['te_hist_true'],g['te_hold_true']
    above = (np.abs(hist)>d['yield_stress']).any(axis=1) | (np.abs(hold)>d['yield_stress'])
    slices = {'all': np.ones(len(labs),bool), 'complete': ~missing, 'missing': missing, 'above': above, 'below': ~above}
    slices.update({f'lab {i}': labs==i for i in range(4)})
    slices.update({f'wing {i}': labs//2==i for i in range(2)})
    for path in ('/tmp/poly_replay.py','/holdout_task/alpha_w01_batch_08_solver.py'):
        p=load(path)(d); ec=p['predicted_strain']-g['te_clean']; er=p['initial_rate_pred']-g['te_true_rate0']; print(path)
        for k,j in [('estimated_taus','taus'),('hereditary_weights','theta'),('viscoplastic_weights','vp_theta')]:
            print(k,'estimate',p[k].round(7),'reference',np.asarray(g[j]).round(7),'max abs error',max(abs(p[k]-g[j])))
        for key, rows in slices.items():
            if rows.any(): print(key, rows.sum(), 'curve/rate RMSE',round(rm(ec[rows]),7),round(rm(er[rows]),7))
