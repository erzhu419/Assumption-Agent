import copy, importlib.util, pickle
import numpy as np

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m.solve_viscoplastic_hereditary_revision

root='/holdout_task/'
old=load('old','/tmp/vp_baseline.py')
new=load('new','/tmp/vp_rate_all.py')
case=pickle.load(open(root+'public_suc01_C_datasets.pkl','rb'))[0]
d,g=case['dataset'],case['gt']; te=d['test']; y=d['yield_stress']
over=(abs(te['stress_hold_0'])>y)|(abs(te['past_stress'])>y).any(1)
rng=np.random.default_rng(564)
for scenario in ('public','hide_one_per_coupon','hide_two_per_coupon'):
    ds=copy.deepcopy(d)
    if scenario!='public':
        x=ds['test']['coupon_features_raw']
        for row in x:
            avail=np.flatnonzero(np.isfinite(row)); count=1 if scenario=='hide_one_per_coupon' else 2
            row[rng.choice(avail,min(count,len(avail)),replace=False)]=np.nan
    missing=~np.isfinite(ds['test']['coupon_features_raw']).all(1)
    for name,solve in [('baseline',old),('rate-corrected',new)]:
        p=solve(ds); e=abs(p['predicted_strain']-g['te_clean']); taus=p['estimated_taus']; vp=p['viscoplastic_weights']
        metrics=[e.mean(),e[over].mean(),e[~over].mean(),e[missing].mean(),e[missing,-1].mean()]
        print(scenario,name,'MAE all/over/sub/missing/missing-late',np.round(metrics,6))
        if name=='rate-corrected':
            assert list(taus)==sorted(taus) and 0.2<=taus[0]<taus[1]<=8
            assert np.max(abs(taus/g['taus']-1))<=.10
            assert np.linalg.norm(vp-g['vp_theta'])/np.linalg.norm(g['vp_theta'])<=.14
            assert np.all(np.asarray(metrics[:3])<= [.070,.080,.065])
