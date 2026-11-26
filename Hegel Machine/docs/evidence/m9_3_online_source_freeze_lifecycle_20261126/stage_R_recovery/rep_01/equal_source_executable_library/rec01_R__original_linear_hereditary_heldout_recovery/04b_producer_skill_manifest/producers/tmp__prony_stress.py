import importlib.util,pickle
import numpy as np
spec=importlib.util.spec_from_file_location('challenger','/tmp/prony_challenger.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
r=pickle.load(open('/holdout_task/public_rec01_R_datasets.pkl','rb'))[0]
d=r['dataset']; g=r['gt']; test=d['test']; train=d['train']; t=d['eval_times']; dt=d['past_dt']
rmse=lambda e: np.mean(e**2)**.5
print('Unseen-lab pooling: compare rate anchor with mild weight on transferred coupon calibration')
for lab in np.unique(train['lab_id']):
    sub={k:v[train['lab_id']!=lab] for k,v in train.items()}
    o=m.solve_hereditary_prony_recovery({**d,'train':sub})
    rows=test['lab_id']==lab; full=rows & np.isfinite(test['coupon_features_raw']).all(axis=1)
    panel=np.concatenate([s['ref_coupons'][s['lab_id']==lab] for s in (train,test)])
    x=(test['coupon_features_raw'][full]-np.nanmedian(np.nanmean(panel,axis=0)))/np.nanmedian(np.nanstd(panel,axis=0))
    _,b=m._basis(test,dt,t,o['estimated_taus']); theta=o['hereditary_weights']
    target=b[full]@theta[:2]+theta[2]+x@theta[3:]
    for weight in (0.,.1,.35):
        rate=o['initial_rate_pred'].copy(); rate[full]+=weight*(target-rate[full]); curve=o['predicted_strain']+(rate-o['initial_rate_pred'])[:,None]*t
        print(lab,weight,round(rmse(curve[rows]-g['te_clean'][rows]),6),round(rmse(rate[rows]-g['te_true_rate0'][rows]),6))
print('Synthetic variable-step replay: truth, fit, slow/fast max relative error, strain and rate RMSE')
for k,truth in enumerate(([.2,8.],[.23,1.4],[1.5,7.8],[.70,1.05],[2.0,2.4])):
    rng=np.random.default_rng(875+k); altered=dt*rng.uniform(.45,1.7,dt.size); tt=t*rng.uniform(.9,1.1,t.size); coef=np.array([.65,.5]); truth=np.array(truth)
    copies=[]; exact=[]
    for split in (train,test):
        n=len(split['lab_id']); x=rng.normal(size=(n,4)); ids=split['lab_id']; raw=x*np.array([1.32,.79,1.41,.85])[ids,None]+np.array([3.1,-3.4,3.6,-2.9])[ids,None]; raw[~np.isfinite(split['coupon_features_raw'])]=np.nan
        a,b=m._basis(split,altered,tt,truth); drift=.08+x@np.array([-.15,.12,.10,-.09]); curves=split['strain_0'][:,None]+a@coef+drift[:,None]*tt; rates=b@coef+drift
        y=curves+rng.normal(0,.014,curves.shape); glitches=rng.random(curves.shape)<.08; y[glitches]+=.35*rng.standard_cauchy(glitches.sum())
        copies.append({**split,'coupon_features_raw':raw,'strain_curves_obs':y,'instant_rate_obs':rates+rng.normal(0,.01,n)}); exact.append((curves,rates))
    o=m.solve_hereditary_prony_recovery({'train':copies[0],'test':copies[1],'past_dt':altered,'eval_times':tt}); taus=o['estimated_taus']; y,v=exact[1]
    print(truth,taus,round(max(abs(taus/truth-1)),4),round(rmse(o['predicted_strain']-y),5),round(rmse(o['initial_rate_pred']-v),5))
