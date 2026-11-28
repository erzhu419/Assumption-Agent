import importlib.util,pickle,numpy as np
p='/holdout_task/'; spec=importlib.util.spec_from_file_location('solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.load(open(p+'public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; raw=c['bridge_raw']; true=c['true_channel_loads']; clean=np.max(abs(raw),axis=1)<5
q=[]
for i,idx in enumerate(np.array_split(np.random.default_rng(42).permutation(np.flatnonzero(clean)),5)):
    keep=np.ones(len(raw),bool); keep[idx]=False
    b=m._bridge({**c,'bridge_raw':raw[keep],'true_channel_loads':true[keep]}); pred=m._loads(raw[idx],b)
    err=np.abs((pred-true[idx]).sum(axis=-1)); q.append(np.quantile(err,.9)); print('bridge fold',i,'stress p90/max',q[-1],err.max())
assert max(q)<.005
b=m._bridge(c); o,gain,a,f,k,v,h=b
print('bridge parameters',b,'diagonal dominance margin',np.min(1-abs(a*f))-abs(k)-abs(v*h))
for name in ('train','test'):
    for key in ('past_channel_raw','hold_channel_raw'):
        orig=d[name][key]; x=m._loads(orig,b); w=x[...,m._OPPOSITE]
        fitted=o+gain*(x+a*np.sin(f*x)+k*w+v*(1-np.cos(h*w)))
        assert np.max(abs(fitted-orig))<1e-9
tr,te=d['train'],d['test']; t,dt=d['eval_times'],d['past_dt']; x,_=m._features(tr,te)
y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; samples=m._prepare(tr,x,b)
out=m.solve_batch(d); print('full material',out['estimated_taus'],out['hereditary_weights']); assert .05<out['estimated_taus'][0]<out['estimated_taus'][1]<25
for lab in np.unique(tr['lab_id']):
    k=tr['lab_id']!=lab; u,w=m._fit(tuple(v[k] for v in samples),dt,t,y[k],r[k]); g,h=m._design(np.log(u),tuple(v[~k] for v in samples),dt,t); curve=np.abs(g@w-y[~k]); rate=np.abs(h@w-r[~k]); print('held lab',lab,'tau',u,'early/late curve p90',*[round(np.quantile(z,.90),6) for z in np.array_split(curve,2,axis=1)],'rate p90',round(np.quantile(rate,.90),6))
