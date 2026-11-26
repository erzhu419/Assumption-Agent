import importlib.util
import pickle
import sys
import numpy as np

path = sys.argv[1]
sp = importlib.util.spec_from_file_location('solver', path)
s = importlib.util.module_from_spec(sp)
sp.loader.exec_module(s)
solver = getattr(s, 'solve_alpha_w01_batch_' + ('05' if '05' in path else '04'))
d = pickle.load(open('/holdout_task/public_alpha_w01_batch_05.pkl', 'rb'))[0]['dataset']
tr = d['train']
lab = tr['lab_id']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
curve = np.empty_like(tr['strain_curves_obs'])
rates = np.empty_like(tr['instant_rate_obs'])
blind_curve = np.empty_like(curve)
blind_rate = np.empty_like(rates)
for i in np.unique(lab):
    v = lab == i
    train = {k: a[~v] for k, a in tr.items()}
    test = {k: a[v] for k, a in tr.items() if k != 'strain_curves_obs'}
    ans = solver(dict(d, train=train, test=test))
    blind = solver(dict(d, train=train, test={k: a for k, a in test.items() if k != 'instant_rate_obs'}))
    curve[v], rates[v] = ans['predicted_strain'], ans['initial_rate_pred']
    blind_curve[v], blind_rate[v] = blind['predicted_strain'], blind['initial_rate_pred']
    print('held lab', i, 'taus', ans['estimated_taus'].round(5), 'theta', ans['hereditary_weights'].round(5), 'plastic', ans['viscoplastic_weights'])

def stats(err):
    return np.r_[np.sqrt(np.mean(err**2)), np.median(np.abs(err)), np.quantile(np.abs(err), .9)].round(5).tolist()

err = curve - tr['strain_curves_obs']
be = blind_curve - tr['strain_curves_obs']
br = blind_rate - tr['instant_rate_obs']
for i in np.unique(lab):
    for c in (True, False):
        v = (lab == i) & (complete == c)
        print('slice', i, 'complete', c, 'n',sum(v),'curve_rmse/MedAE/P90',stats(err[v]),'blind curve',stats(be[v]),'blind rate',stats(br[v]))
print('blind vs informed missing curves', stats(be[~complete]), stats(err[~complete]))
print('complete curves per horizon quantiles', np.quantile(abs(err[complete]),[.5,.8,.9,.95,.99],axis=0).round(4))
print('complete curve/rate signed residual quantiles', np.quantile(err[complete],[0,.01,.05,.5,.95,.99,1]).round(4), np.quantile(br[complete],[0,.01,.05,.5,.95,.99,1]).round(4))
print('complete within-coupon end-to-end residual correlation', np.corrcoef(err[complete,0],err[complete,-1])[0,1])
