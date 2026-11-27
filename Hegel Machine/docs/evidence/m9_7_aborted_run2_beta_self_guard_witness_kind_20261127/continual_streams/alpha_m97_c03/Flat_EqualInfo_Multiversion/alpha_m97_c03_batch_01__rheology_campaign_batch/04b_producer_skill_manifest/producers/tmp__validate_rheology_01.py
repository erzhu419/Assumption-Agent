import importlib.util, pickle, time
import numpy as np
root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_m97_c03_batch_01_solver.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
item = pickle.load(open(root+'public_alpha_m97_c03_batch_01.pkl','rb'))[0]
d = item['dataset']; tr, te = d['train'], d['test']
start=time.monotonic(); out=s.solve_alpha_m97_c03_batch_01(d)
print('fit secs',round(time.monotonic()-start,3), 'fit', {k:np.round(out[k],7).tolist() for k in ('estimated_taus','hereditary_weights','regime_mode_weights')})
gt=item['gt']; print('public evaluation fields', {k:np.shape(v) for k,v in gt.items()})
for k in out.keys() & gt.keys():
    err = out[k]-gt[k]
    print('public',k,'RMSE',np.sqrt(np.mean(err**2)),'max abs',np.max(np.abs(err)))
cal=d['calibration_load_cells']; g,b=s._calibrate(cal)
spike={k:v.copy() for k,v in cal.items()}; spike['true_stress'][2] += 7
sg,sb=s._calibrate(spike); assert max(np.max(abs(sg-g)),np.max(abs(sb-b))) < .001
sparse={k:v[(cal['lab_id'] != 1) | (np.arange(len(cal['lab_id'])) == 10)] for k,v in cal.items()}
ag,ab=s._calibrate(sparse); assert abs(ag[1]-1) < .001 and abs(ab[1]) < .001
assert np.allclose(ag[[0,2,3]],g[[0,2,3]]) and np.allclose(ab[[0,2,3]],b[[0,2,3]])
print('EMI and sparse archive paths pass')
x,_=s._align(tr,te)
args = g,b,d['past_dt'],d['eval_times'],d['crit_stress'],d['shear_banding_index']
hist = g[tr['lab_id'],None]*tr['past_stress_cmd']+b[tr['lab_id'],None]
damage=np.maximum(abs(hist)-d['crit_stress'],0)**2 @ d['past_dt']
cv=[]
for lab in range(4):
    m=tr['lab_id']==lab
    train={k:v[~m] for k,v in tr.items()}; test={k:v[m] for k,v in tr.items()}
    build,_=s._design(train,x[~m],*args)
    logtau,beta=s._fit(build,train['strain_curves_obs']-train['strain_0'][:,None],train['instant_rate_obs'],args[2],args[3])
    build,arrest=s._design(test,x[m],*args); c,r=build(logtau)
    ce=c@beta+test['strain_0'][:,None]-test['strain_curves_obs']; re=r@beta-test['instant_rate_obs']
    for label,mask in [('all',np.ones(m.sum(),bool)),('inactive',~arrest),('weak',arrest&(damage[m]<1)),('strong',arrest&(damage[m]>=1)),('missing',np.isnan(test['coupon_features_raw']).any(axis=1))]:
        if mask.any():
            cv.append((lab,label,int(mask.sum()),round(np.sqrt(np.mean(ce[mask]**2)),6),round(np.sqrt(np.mean(re[mask]**2)),6)))
    print('held lab',lab,'taus',np.round(np.exp(logtau),6),'regime weights',np.round(beta[7:],6))
print('LOLO (lab,slice,n,strain RMSE,rate RMSE)',cv)
