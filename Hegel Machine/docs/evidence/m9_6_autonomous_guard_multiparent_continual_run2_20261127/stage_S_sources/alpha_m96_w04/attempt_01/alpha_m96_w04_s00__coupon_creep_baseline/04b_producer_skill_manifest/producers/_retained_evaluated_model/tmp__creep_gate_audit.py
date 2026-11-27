import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_m96_w04_s00_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_m96_w04_s00.pkl', 'rb') as f:
    case = pickle.load(f)[0]
d,g=case['dataset'],case['gt']; tr,te=d['train'],d['test']; t=d['eval_times']; n=len(tr['lab_id'])
out=s.solve_alpha_m96_w04_s00(d)
print('reference panel shapes',tr['ref_coupons'].shape,te['ref_coupons'].shape)
print('test labels/measurements', sorted(te))
c=np.column_stack((g['stress_gains'],g['stress_biases'])); x=s._aligned((tr,te))[n:]; w=g['theta']
base=s._Model(te,x,c,d['past_dt'],t).basis(g['taus'])[0][...,:2]@w[:2]
true_drift=g['te_true_rate0']-base[:,-1]
truth=g['te_clean']-te['strain_0'][:,None]
print('reference branch closure max abs',np.max(abs(truth-base[:,:-1]-true_drift[:,None]*t)))
raw=np.concatenate((tr['coupon_features_raw'],te['coupon_features_raw'])); ref=np.concatenate((tr['ref_coupons'],te['ref_coupons'])); lab=np.r_[tr['lab_id'],te['lab_id']]
complete=np.isfinite(raw[n:]).all(1)
for pooling in ('joint','train'):
 for convention in ('median','column','pooled'):
    z=raw.copy()
    for k in np.unique(lab):
        m=lab==k; panel=ref[m & (np.arange(len(lab))<n)] if pooling=='train' else ref[m]
        loc=np.nanmean(panel,axis=0); scale=np.nanstd(panel,axis=0)
        if convention=='median': loc,scale=np.nanmedian(loc),np.nanmedian(scale)
        if convention=='pooled': loc,scale=np.nanmean(panel),np.nanstd(panel)
        if np.shape(loc) and np.shape(loc) != (raw.shape[1],): break
        z[m]=(z[m]-loc)/scale
    else:
        e=(w[2]+z[n:]@w[3:]-true_drift)[complete]
        print('complete drift oracle RMSE',pooling,convention,np.sqrt(np.mean(e*e)))
for k in np.unique(tr['lab_id']):
    m=tr['lab_id']==k; split=dict(d,train={a:v[~m] for a,v in tr.items()},test={a:v[m] for a,v in tr.items()}); p=s.solve_alpha_m96_w04_s00(split)
    keep=np.isfinite(tr['coupon_features_raw'][m]).all(1)
    e=(tr['instant_rate_obs'][m]-p['initial_rate_pred'])[keep]
    r=(tr['strain_curves_obs'][m]-p['predicted_strain'])[keep]; slope=np.median(r/t,axis=1)
    print('held-out lab',k,'complete onset median signed/MAD',np.median(e),s._mad(e),'curve drift median signed/MAD',np.median(slope),s._mad(slope))
for name,mask in [('complete',complete),('missing',~complete)]:
    print(name,'observed test onset versus clean RMSE',np.sqrt(np.mean((te['instant_rate_obs'][mask]-g['te_true_rate0'][mask])**2)), 'solver clean onset RMSE',np.sqrt(np.mean((out['initial_rate_pred'][mask]-g['te_true_rate0'][mask])**2)))
