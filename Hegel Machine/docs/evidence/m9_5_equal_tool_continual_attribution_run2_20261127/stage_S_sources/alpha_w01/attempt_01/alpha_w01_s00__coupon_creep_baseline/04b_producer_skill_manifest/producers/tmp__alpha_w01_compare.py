import sys, pickle, numpy as np
sys.path.insert(0, '/holdout_task')
import alpha_w01_s00_solver as s
row = pickle.load(open('/holdout_task/public_alpha_w01_s00.pkl','rb'))[0]
d, gt = row['dataset'], row['gt']
def run(d, fit_complete, correction):
    tr, te = d['train'], d['test']
    dt,t=d['past_dt'],d['eval_times']
    maps=s._load_cells(d['calibration_load_cells'])
    x,xt=s._alignment(tr,te)
    m=np.isfinite(tr['coupon_features_raw']).all(axis=1) if fit_complete else np.ones(len(x),bool)
    taus,w=s._fit({k:v[m] for k,v in tr.items()},x[m],maps,dt,t)
    a,b=s._design(taus,*s._stress(te,maps),np.c_[np.ones(len(xt)),xt],dt,t)
    ep=te['strain_0'][:,None]+a@w
    rate=b@w
    missing=~np.isfinite(te['coupon_features_raw']).all(axis=1)
    delta=np.where(missing,te['instant_rate_obs']-rate,0)*correction
    return taus,w,ep+delta[:,None]*t,rate+delta

def report(label, te, e, r):
    miss=~np.isfinite(te['coupon_features_raw']).all(axis=1)
    mae=lambda z: np.mean(abs(z))
    print(label,'strain/rate MAE',[(round(mae(e[m]),5),round(mae(r[m]),5)) if m.any() else None for m in (np.ones(len(e),bool),miss,~miss)],'early/late',np.round(np.median(abs(e),axis=0)[[0,-1]],5))
for fit in (False,True):
    taus,w,p,q=run(d,fit,0.)
    print('complete fit',fit,'tau error',taus-gt['taus'],'weight error norm',np.linalg.norm(w-gt['theta']))
    for c in (0.,1.):
        _,_,p,q=run(d,fit,c)
        report(('clean',fit,c),d['test'],p-gt['te_clean'],q-gt['te_true_rate0'])
tr=d['train']
for lab in np.unique(tr['lab_id']):
    m=tr['lab_id']==lab
    dd=dict(d, train={k:v[~m] for k,v in tr.items()},test={k:v[m] for k,v in tr.items() if k!='strain_curves_obs'})
    for fit,c in ((False,0.),(False,1.),(True,1.)):
        ts,w,p,q=run(dd,fit,c)
        report(('LOLO',lab,fit,c),dd['test'],p-tr['strain_curves_obs'][m],q-tr['instant_rate_obs'][m])
