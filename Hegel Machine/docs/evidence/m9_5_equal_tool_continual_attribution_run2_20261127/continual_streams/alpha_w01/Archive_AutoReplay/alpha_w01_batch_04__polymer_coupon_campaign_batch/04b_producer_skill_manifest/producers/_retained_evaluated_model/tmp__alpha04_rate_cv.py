import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_w01_batch_04_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
d = pickle.load(open(root+'public_alpha_w01_batch_04.pkl','rb'))[0]['dataset']
b = d['train']; labs = b['lab_id']; t = d['eval_times']; missing = ~np.isfinite(b['coupon_features_raw']).all(axis=1)

def sub(mask, drop=()):
    return {k: np.asarray(v)[mask] for k, v in b.items() if k not in drop}

for labs_out in ([0],[1],[2],[3],[0,1],[2,3]):
    v = np.isin(labs,labs_out)
    test = sub(v, ('strain_curves_obs','instant_rate_obs'))
    fit = s.solve_alpha_w01_batch_04({**d,'train':sub(~v),'test':test})
    e = fit['predicted_strain'] - b['strain_curves_obs'][v]
    r = b['instant_rate_obs'][v] - fit['initial_rate_pred']
    m = missing[v]
    rate_to_curve = (e[m] @ t)/(t @ t)
    print('held',labs_out,'missing rates and curve-implied deficits:',np.round(np.c_[r[m], -rate_to_curve],5).tolist())
    for a in (0., .5, .9, 1.):
        adj = e[m] + a*r[m,None]*t
        print(a, np.round([np.abs(adj).mean(), np.sqrt((adj**2).mean()), np.abs(adj[:,-1]).mean()],5))
    # Simulated missing single features test whether the rate correction also helps otherwise-complete coupons.
    ids = np.flatnonzero(~m)
    altered = test['coupon_features_raw'].copy()
    altered[ids,np.arange(len(ids))%4]=np.nan
    changed = s.solve_alpha_w01_batch_04({**d,'train':sub(~v),'test':{**test,'coupon_features_raw':altered}})
    e = changed['predicted_strain'][ids] - b['strain_curves_obs'][v][ids]
    r = b['instant_rate_obs'][v][ids] - changed['initial_rate_pred'][ids]
    print('masked median curve abs errors', [round(float(np.median(np.abs(e+a*r[:,None]*t))),5) for a in (0,.5,.9,1)])
out = s.solve_alpha_w01_batch_04(d)
test=d['test']; m=~np.isfinite(test['coupon_features_raw']).all(1)
r=test['instant_rate_obs']-out['initial_rate_pred']; print('public test true missing residuals',r[m].round(5),'complete absolute residual quantiles',np.quantile(np.abs(r[~m]),[.5,.9,1]))
