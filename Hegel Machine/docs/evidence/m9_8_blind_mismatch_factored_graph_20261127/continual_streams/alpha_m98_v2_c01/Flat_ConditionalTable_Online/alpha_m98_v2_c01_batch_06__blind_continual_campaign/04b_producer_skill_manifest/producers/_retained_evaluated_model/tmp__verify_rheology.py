import importlib.util, pickle, time
import numpy as np
spec=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; tr,te,g=d['train'],d['test'],i['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; labs=tr['lab_id']; glabs=g['lab_id']
start=time.monotonic(); pred=m.solve_batch(d); tau,w,C,B=(pred[k] for k in ('estimated_taus','hereditary_weights','predicted_strain','initial_rate_pred')); print('seconds',round(time.monotonic()-start,3),'tau',tau,'theta',w)
assert [v.shape for v in (tau,w,C,B)]==[(2,),(7,),(len(te['strain_0']),len(t)),(len(te['strain_0']),)]
assert all(np.isfinite(v).all() for v in (tau,w,C,B)) and .05<tau[0]<tau[1]<25
x,u=m._features(tr,te); gx=m._features(g,g)[0]; bridge=m._axial_load_map(d['calibration_load_cells']); src=m._specimen(tr,x,bridge); _,th,tf=m._specimen(te,u,bridge); _,gh,gf=m._specimen(g,gx,bridge)
zero,rate=m._chamber_increment(th,tf,np.array([0.,1e-7]),tau,w); assert np.array_equal(zero[:,0],np.zeros(len(th))) and np.allclose(zero[:,1]/1e-7,B,atol=1e-6,rtol=1e-6) and np.allclose(rate,B)
obs=tr['strain_curves_obs']-tr['strain_0'][:,None]; robs=tr['instant_rate_obs']; go=g['strain_curves_obs']-g['strain_0'][:,None]; gr=g['instant_rate_obs']; missing=np.isnan(g['coupon_features_raw']).any(1)
def rmse(curve,rate): return np.sqrt([np.mean(curve**2),np.mean(rate**2)])
cs,rs=m._chamber_increment(gh,gf,t,tau,w); e,f=cs-go,rs-gr
print('chamber fixed law pooled/complete/missing',rmse(e,f),rmse(e[~missing],f[~missing]),rmse(e[missing],f[missing])); print('chamber first/last horizon RMSE',np.sqrt(np.mean(e[:,[0,-1]]**2,axis=0)))
train_scores=[]; chamber_scores=[]
for lab in np.unique(labs):
 a=labs!=lab; sp=tuple(v[a] for v in src); tv,wv=m._fit(sp,dt,t,obs[a],robs[a]); v=tuple(x[~a] for x in src); cb,rb=m._branches(v,dt,t,tv); q=v[2]@wv[2:]; b0=cb@wv[:2]+q[:,None]*t; b1=rb@wv[:2]+q
 j=glabs==lab; cg,rg=m._chamber_increment(gh[j],gf[j],t,tv,wv); train_scores.append(np.median(abs(b0-obs[~a]))); chamber_scores.append(rmse(cg-go[j],rg-gr[j])); print('held-out lab',lab,'tau',tv,'train median abs strain/rate',train_scores[-1],np.median(abs(b1-robs[~a])),'chamber RMSE',chamber_scores[-1])
print('worst held-out chamber lab',np.max(chamber_scores,axis=0))
cal=d['calibration_load_cells']; raw=cal['bridge_raw']; loads=cal['true_channel_loads']; good=np.max(abs(raw-np.median(raw,axis=0)),axis=1)<8; err=[]
for fold in range(7):
 held=np.arange(len(raw))%7==fold; fit=m._axial_load_map(dict(bridge_raw=raw[~held],true_channel_loads=loads[~held])); held &= good; err.extend((m._channel_loads(raw[held],fit)-loads[held]).sum(axis=-1))
print('held-out bridge axial RMSE/max',np.sqrt(np.mean(np.square(err))),max(np.abs(err)))
