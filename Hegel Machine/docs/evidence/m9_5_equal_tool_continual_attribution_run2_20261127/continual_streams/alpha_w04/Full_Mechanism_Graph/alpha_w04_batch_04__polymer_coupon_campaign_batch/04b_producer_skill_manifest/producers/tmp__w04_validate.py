import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_w04_batch_04_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for record in pickle.load(open(root+'public_alpha_w04_batch_04.pkl','rb')):
    d = record['dataset']
    o = m.solve_alpha_w04_batch_04(d)
    print('fit', {k: np.round(v,7).tolist() for k,v in o.items() if v.size < 10})
    gt = record['gt']
    print('public reference fields', [(k,np.shape(v)) for k,v in gt.items()] if isinstance(gt,dict) else type(gt).__name__)
    if isinstance(gt,dict):
        for k,v in o.items():
            if k in gt and np.shape(gt[k]) == v.shape:
                e = np.asarray(v)-np.asarray(gt[k]); print('public reference',k,'RMSE',np.sqrt(np.mean(e**2)),'max',np.max(np.abs(e)))
    tr=d['train']; labs=tr['lab_id']; n=len(labs); obs=tr['strain_curves_obs']; rate=tr['instant_rate_obs']; dt=d['past_dt']; t=d['eval_times']; y=d['yield_stress']
    c=d['calibration_load_cells']; affine=m._calibrate(c)
    x=m._align(tr,d['test'])[0]; a=m._inputs(tr,affine,x,dt,y)
    cv=np.empty_like(obs); rv=np.empty_like(rate)
    for lab in range(4):
        keep=labs!=lab; take=~keep
        fold=tuple(z[keep] for z in a); active=tuple(bool(np.any(z)) for z in fold[-2:]); tt,ww=m._fit(dt,t,fold,obs[keep]-tr['strain_0'][keep,None],rate[keep],active)
        ca,rb=m._design(tt,dt,t,tuple(z[take] for z in a),active)
        cv[take]=tr['strain_0'][take,None]+ca@ww; rv[take]=rb@ww
        print('fold',lab,'taus',tt.round(5),'plastic',ww[7:].round(6))
    categories=(np.abs(a[3])>0).astype(int)+2*(np.max(np.abs(a[0]),axis=1)>y)
    def slice_scores(mask):
        return tuple(float(np.sqrt(np.mean(e[mask]**2))) for e in (cv-obs,rv-rate))
    metrics=[(lab,rg,int(mask.sum()),*slice_scores(mask)) for lab in range(4) for rg in range(4) if (mask:=(labs==lab)&(categories==rg)).any()]
    print('lab x yield regime LOLO RMSE [lab, regime, n, curve, rate]',metrics)
    print('worst LOLO curve,rate',max(s[3] for s in metrics),max(s[4] for s in metrics))
    for lab in range(4):
        rows=np.flatnonzero(c['lab_id']==lab); xcal=c['cmd_stress'][rows]; keep=rows[[xcal.argmin(),xcal.argmax()]]
        mask=(c['lab_id']!=lab)|np.isin(np.arange(len(c['lab_id'])),keep)
        sparse={k:v[mask].copy() for k,v in c.items()}; sparse['true_stress'][np.flatnonzero(sparse['lab_id']==lab)[0]]+=0.16
        recovered=m._calibrate(sparse)
        dirty={k:v.copy() for k,v in c.items()}; dirty['true_stress'][rows[:2]] += [0.16,-0.24]
        robust=m._calibrate(dirty)
        print('calibration stress test',lab,'sparse + EMI',np.max(np.abs(recovered-affine)), 'dense + EMI',np.max(np.abs(robust-affine)))
    sub=dict(d,yield_stress=1e6); so=m.solve_alpha_w04_batch_04(sub)
    assert np.array_equal(so['viscoplastic_weights'],np.zeros(2))
    print('sub-yield branch weights exactly zero')
