import importlib.util, pickle, numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m97_c02_batch_01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb'))[0]; d=i['dataset']; tr,te=d['train'],d['test']; c=d['calibration_load_cells']; full=m._calibration(d); x=m._align(d)[:len(tr['lab_id'])]; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; gt=i['gt']
for lab in range(4):
    rows=np.flatnonzero(c['lab_id']==lab); sel=te['lab_id']==lab; other=np.arange(4)!=lab
    for count in (0,1):
        keep=c['lab_id']!=lab; keep[rows[:count]]=True
        q=dict(d,calibration_load_cells={k:v[keep] for k,v in c.items()}); sparse={}; cal=m._calibration(q,sparse)
        m._refine_calibration(q,cal,sparse,x,y,r)
        err=np.max(np.abs(cal[lab]-full[lab])); assert err < .013 and np.array_equal(cal[other],full[other])
        out=m.solve_alpha_m97_c02_batch_01(q)
        rmse=[np.sqrt(np.mean((out[k][sel]-gt[g][sel])**2)) for k,g in [('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]]
        assert rmse[0]<.10 and rmse[1]<.08
        print(lab,count,'calibration max error',round(err,5),'strain/rate RMSE',np.round(rmse,5))
