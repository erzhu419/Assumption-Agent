exec(open('/tmp/coupon04_validate.py').read().split('res = np.empty_like(y)')[0])
miss = np.isnan(tr['coupon_features_raw']).any(axis=1)
print('Test onset finite',np.isfinite(te['instant_rate_obs']).sum(), 'test missing with onset',np.sum(np.isnan(te['coupon_features_raw']).any(axis=1)&np.isfinite(te['instant_rate_obs'])))
for c in range(4):
    learn = lab != c
    a = {k:v[learn] for k,v in tr.items()}
    tau0,w0 = s._fit(s._Design(d,a,x[learn],cal),a)
    fit = learn & ~miss
    b = {k:v[fit] for k,v in tr.items()}
    tau,w = s._fit(s._Design(d,b,x[fit],cal),b)
    prior = full.make(tau0) @ w0
    pred = full.make(tau) @ w
    delta = y[:,-1] - pred[:,-1]
    delta = np.where(miss & np.isfinite(delta),delta,0.)
    corrections = [np.column_stack((q[:,None]*full.t, q)) for q in (delta, np.clip(delta,-.35,.35), .8*np.clip(delta,-.35,.35))]
    print('lab',c,'complete-only fit',tau,'missing onset deltas',np.round(delta[(lab==c)&miss],4))
    for name, v in [('original',prior),('complete-fit',pred),*[(f'correction {i}',pred+z) for i,z in enumerate(corrections)]]:
        for m in ((lab==c)&miss, (lab==c)&~miss):
            err = (v-y)[m,:-1]
            print(name,len(err),np.round([np.sqrt(np.nanmean(err**2)), np.nanmedian(abs(err)), np.nanmedian(abs(err[:,-1]))],6))
item = pickle.load(open('/holdout_task/public_alpha_m96_w02_batch_04.pkl','rb'))[0]
ref = item['gt']; mt = np.isnan(te['coupon_features_raw']).any(axis=1)
print('test onset reference error, missing rows',np.round((te['instant_rate_obs']-ref['te_true_rate0'])[mt],6),'complete RMSE',np.sqrt(np.mean((te['instant_rate_obs'][~mt]-ref['te_true_rate0'][~mt])**2)))
