import importlib.util,pickle,numpy as np
root='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',root+'preloaded_hereditary_secant_hazard_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def errors(e, groupings):
    print('pooled',np.round(e.mean(0),6))
    for name, groups in groupings:
        vals = [(g, e[groups == g].mean(0)) for g in np.unique(groups)]
        print(name,[(str(g),np.round(v,6).tolist()) for g,v in vals] if len(vals)<5 else np.round(np.max([v for g,v in vals],axis=0),6))

def check_readouts(data,out,t):
    y = out['predicted_strain']; y0 = data['strain_0']; crit = data['critical_strain']; sec=out['secant_compliance']; cross=out['hazard_crossing_times']
    np.testing.assert_allclose(sec,(y[:,-1]-y0)/np.maximum(abs(data['stress_hold_0']),.5),rtol=0,atol=1e-13)
    manual=[]
    for vals,target in zip(np.column_stack((y0,y)),crit):
        j=next((j for j,v in enumerate(vals) if v>=target),None)
        grid=np.r_[0.,t]
        manual.append(t[-1] if j is None else 0. if j==0 else np.interp(target,vals[j-1:j+1],grid[j-1:j+1]))
    np.testing.assert_allclose(cross,manual,rtol=0,atol=1e-12)

with open(root+'public_suc01_T_datasets.pkl','rb') as f: items=pickle.load(f)
limits=np.array([.075,.095,.28])
for item in items:
    d,g=item['dataset'],item['gt']; tr,te=d['train'],d['test']; t,dt=d['eval_times'],d['past_dt']; out=m.solve_preloaded_hereditary_transfer(d)
    check_readouts(te,out,t)
    tauerr=abs(out['estimated_taus']/g['taus']-1); print('public tau relative errors',tauerr)
    e=np.column_stack([abs(out['predicted_strain']-g['te_clean']).mean(1),abs(out['secant_compliance']-g['te_secant']),abs(out['hazard_crossing_times']-g['te_cross'])])
    print('public clean [strain, secant, crossing]')
    errors(e,[('lab',te['lab_id']),('stratum',te['stratum']),('lab-stratum worst',3*te['lab_id']+te['stratum']),('missing coupon',~np.isfinite(te['coupon_features_raw']).all(1))])
    assert np.max(tauerr) <= .10 and np.all(e.mean(0) <= limits)
    print('public clean per-horizon strain MAE:', np.round(abs(out['predicted_strain']-g['te_clean']).mean(0),6))
    for lab in np.unique(tr['lab_id']):
        sub=lambda i: {k:v[i] for k,v in tr.items()}
        a=sub(tr['lab_id']!=lab); b=sub(tr['lab_id']==lab); obs=b.pop('strain_curves_obs'); b['critical_strain']=b['strain_0']+.25
        pred=m.solve_preloaded_hereditary_transfer(dict(d,train=a,test=b)); check_readouts(b,pred,t)
        r=abs(pred['predicted_strain']-obs)
        print('left-out lab',lab,'tau relative errors',np.round(abs(pred['estimated_taus']/g['taus']-1),5),'observed curve median error',round(float(np.median(r)),6),'observed curve worst horizon median',round(float(np.max(np.median(r,axis=0))),6))
