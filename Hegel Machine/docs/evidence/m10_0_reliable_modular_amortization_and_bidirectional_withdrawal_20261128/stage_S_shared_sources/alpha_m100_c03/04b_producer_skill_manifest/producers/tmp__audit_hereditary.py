import importlib.util, json, pickle, numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; c=d['calibration_load_cells']; raw,load=c['bridge_raw'],c['true_channel_loads']; truth=load.sum(axis=1)
def cal(a,degree):
    z=[np.ones(len(a)),*a.T]
    if degree==2: z.extend(a[:,i]*a[:,j] for i in range(4) for j in range(i,4))
    return np.column_stack(z)
idx=np.arange(len(raw)); order=np.argsort(truth); fold=np.empty(len(raw),int); fold[order]=idx%5
for degree in (1,2):
    A=cal(raw,degree); e=np.zeros(len(raw))
    for f in range(5):
        m=fold==f; e[m]=A[m]@np.linalg.lstsq(A[~m], truth[~m],rcond=None)[0]-truth[m]
    print('bridge degree',degree,'RMSE',np.sqrt(np.mean(e**2)),'max load-quartile p90',max(np.quantile(abs(e[ids]),.9) for ids in np.array_split(order,4)))
bridge=s._bridge(c); p,h=s._loads(tr,bridge); x=s._features(tr,te)[:len(h)]; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; labs=tr['lab_id']; aux=tr['aux_chamber_telemetry']; e={k:[] for k in ('specified','lab_intercepts','pressure_grammar','auxiliary_grammar')}
for l in np.unique(labs):
    m=labs!=l; tau,theta=s._fit(x[m],p[m],h[m],dt,t,y[m],r[m]); C,R=s._design(tau,x,p,h,dt,t); ey=y-C@theta; er=r-R@theta
    delta=h-p[:,-1]; asym=aux[:,0]-aux[:,1]; cross=aux[:,2]*aux[:,3]
    # Additive drift challengers; every correction contributes t to strain and one to rate.
    Z={'specified':np.empty((len(h),0)), 'lab_intercepts':np.eye(len(np.unique(labs)))[np.searchsorted(np.unique(labs),labs)], 'pressure_grammar':np.c_[delta,delta/(1+h*h),delta*h*p[:,-1]], 'auxiliary_grammar':np.c_[aux,asym/(1+aux[:,2]**2),cross*asym]}
    for key,z in Z.items():
        a=np.zeros(len(h))
        if z.shape[1]:
            z=z/(1e-8+np.sqrt(np.mean(z[m]**2,axis=0)))
            def res(b):
                v=z[m]@b
                return np.r_[((v[:,None]*t-ey[m])/.012).ravel(),(v-er[m])/.008]
            a=z@least_squares(res,np.zeros(z.shape[1]),loss='cauchy',f_scale=2.5).x
        sy=abs(ey[~m]-a[~m,None]*t); sr=abs(er[~m]-a[~m]); e[key].append([max(np.quantile(sy[:,j],.9) for j in np.array_split(np.arange(len(t)),3)),np.quantile(sr,.9),np.median(sy),np.median(sr)])
    print('lab',l,'signed median residual strain/rate',np.median(ey[~m]),np.median(er[~m]))
for key,v in e.items(): print(key,'worst lab-horizon p90 strain / worst lab rate p90',np.max(v,axis=0)[:2].round(6),'mean lab median',np.mean(v,axis=0)[2:].round(6))
for row in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
    a=np.asarray(row['facility_telemetry_trace']); print('facility fields',list(row),'samples',len(a),'offsets',a.mean(0).round(4),'centered RMS',a.std(0).round(4))
print('verification coupons available',pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['campaign_diagnostics'] is not None)
