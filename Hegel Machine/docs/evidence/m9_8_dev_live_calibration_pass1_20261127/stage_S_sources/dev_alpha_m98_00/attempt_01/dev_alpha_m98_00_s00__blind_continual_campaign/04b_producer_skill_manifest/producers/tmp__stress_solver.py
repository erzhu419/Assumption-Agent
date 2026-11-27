import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; tr,te,pi=(p[k] for k in ('train','test','chamber_verification_trials')); dt,t=p['past_dt'],p['eval_times']; c=m._calibration(p['calibration_load_cells']); a,b=[m._inputs(z,x,c) for z,x in zip((tr,te),m._features(tr,te))]; d=m._inputs(pi,m._features(pi,balanced=False)[0],c); tau,th,s=m._material(tr,a,dt,t)
def q(x): return np.sqrt(np.mean(x*x))
y,r=m._predict(d,dt,t,tau,th)
for name, mem, drift, truth in [('control',None,None,[]),('initial','initial',None,[.65,-.4]),('order','order',None,[-1.]),('thermal',None,1.5,[1.,0.,.01,0.]),('combined','initial','quadratic',[.6,-.3,.45,0.,.02,0.])]:
 z=m._chamber_design(d,dt,t,tau,th,mem,drift)@truth
 syn=dict(pi,strain_curves_obs=pi['strain_curves_obs']+z[:,:-1],instant_rate_obs=pi['instant_rate_obs']+z[:,-1]); score=[]
 for lab in np.unique(pi['lab_id']):
  mask=pi['lab_id']==lab
  mm,dd,beta,_=m._pilot_choice(d,syn,dt,t,tau,th,s,np.flatnonzero(~mask)); pred=m._chamber_design(d,dt,t,tau,th,mm,dd)@beta
  obs=np.c_[syn['strain_curves_obs']-pi['strain_0'][:,None]-y,syn['instant_rate_obs']-r]
  score.append([q(obs[mask,:-1]),q((obs-pred)[mask,:-1]),q(obs[mask,-1]),q((obs-pred)[mask,-1])]); print(name,lab,(mm,dd),np.round(score[-1],5))
 print(name,'worst lab strain before/after, rate before/after',np.round(np.max(score,axis=0),5))
# Reciprocal bridge mixing and offsets must not change inferred total stresses.
rng=np.random.default_rng(511); A=np.eye(4)+rng.normal(0,.15,(4,4)); o=rng.normal(0,.25,4)
cal=dict(p['calibration_load_cells']); cal['bridge_raw']=cal['bridge_raw']@A+o
changed=dict(p,calibration_load_cells=cal)
for key, block in [('train',tr),('test',te),('chamber_verification_trials',pi)]:
 changed[key]=dict(block,past_channel_raw=block['past_channel_raw']@A+o,hold_channel_raw=block['hold_channel_raw']@A+o)
base=m.solve_batch(p); other=m.solve_batch(changed)
print('affine bridge gauge maximum by output',{k:float(np.max(abs(base[k]-other[k]))) for k in base})
