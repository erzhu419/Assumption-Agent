import importlib.util,pickle,numpy as np
from sklearn.model_selection import StratifiedKFold
p='/holdout_task/alpha_w04_batch_05_solver.py'
s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
pub=pickle.load(open('/holdout_task/public_alpha_w04_batch_05.pkl','rb'))[0]; d=pub['dataset']; tr=d['train']; te=d['test']; times=d['eval_times']; late=times>=times[len(times)//4]; cal=m._calibrate(d['calibration_load_cells'])

def trial(a,b):
    out=m.solve_alpha_w04_batch_05({**d,'train':a,'test':b})
    w=np.r_[out['hereditary_weights'],out['viscoplastic_weights']]; tau=out['estimated_taus']
    x,xt=m._align(a,b)
    A,B=m._design(tau,d['past_dt'],times,*m._stress(a,cal),x,d['yield_stress'])
    _,BT=m._design(tau,d['past_dt'],times,*m._stress(b,cal),xt,d['yield_stress'])
    sr=a['strain_curves_obs']-a['strain_0'][:,None]-A@w
    ar=a['instant_rate_obs']-B@w; br=b['instant_rate_obs']-BT@w
    missing=~np.isfinite(b['coupon_features_raw']).all(axis=1)
    def correction(panel,z,inn):
        proxy={**panel,'coupon_features_raw':z,'lab_id':panel['lab_id']}
        return m._alignment_drift(a,proxy,x,z,sr,ar,inn,times)
    da=correction(a,x,np.zeros_like(ar)); db=correction(b,xt,np.zeros_like(br)); db[~missing]=0
    old=m._missing_drift(a,b,x,xt,w[3:7],B@w,BT@w)
    new=m._missing_drift(a,b,x,xt,w[3:7],B@w+da,BT@w+db)
    return out,np.vstack((np.zeros(len(db)),db,db+new-old))

def report(label,sub,p,r,delta,y,rtrue):
    miss=~np.isfinite(sub['coupon_features_raw']).all(axis=1)
    e=p[None]-y+delta[:,:,None]*times; z=r[None]-rtrue+delta
    for name,mask in [('missing',miss), *[(f'lab{j}',miss&(sub['lab_id']==j)) for j in range(4)]]:
        if not mask.any(): continue
        v=e[:,mask]; q=z[:,mask]; print(label,name,mask.sum(),'strain RMSE',np.round(np.sqrt(np.mean(v*v,axis=(1,2))),5),'late slope MAE',np.round(np.mean(abs(np.median(v[:,:,late]/times[late],axis=2)),axis=1),5),'rate RMSE',np.round(np.sqrt(np.mean(q*q,axis=1)),5))
for seed in (94,115):
    p=np.empty_like(tr['strain_curves_obs']); r=np.empty_like(tr['instant_rate_obs']); delta=np.zeros((3,len(r)))
    for fit,val in StratifiedKFold(5,shuffle=True,random_state=seed).split(tr['lab_id'],tr['lab_id']):
        o,delta[:,val]=trial({k:v[fit] for k,v in tr.items()},{k:v[val] for k,v in tr.items() if k!='strain_curves_obs'}); p[val]=o['predicted_strain']; r[val]=o['initial_rate_pred']
    report(f'OOF {seed}',tr,p,r,delta,tr['strain_curves_obs'],tr['instant_rate_obs'])
o,delta=trial(tr,te); report('public clean',te,o['predicted_strain'],o['initial_rate_pred'],delta,pub['gt']['te_clean'],pub['gt']['te_true_rate0'])
