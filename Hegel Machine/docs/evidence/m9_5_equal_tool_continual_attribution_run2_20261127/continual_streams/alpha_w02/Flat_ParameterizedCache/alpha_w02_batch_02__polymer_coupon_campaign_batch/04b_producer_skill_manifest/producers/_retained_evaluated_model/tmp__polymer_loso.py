import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import alpha_w02_batch_02_solver as m
entry=pickle.load(open('/holdout_task/public_alpha_w02_batch_02.pkl','rb'))[0]; d=entry['dataset']; t=d['train']; dt=d['past_dt']; times=d['eval_times']
print('public reference keys',list(entry['gt']))
for k,v in entry['gt'].items():
 if 'hazard' in k or 'cross' in k:
  a=np.asarray(v); print(k,'range',a.min(),a.max(),'outside hold',sum(a>times[-1]),'at zero',sum(a==0),'on grid',sum(np.isin(a,times)))
a=tuple(v[:len(t['lab_id'])] for v in m._inputs(d)); y=np.c_[t['strain_curves_obs']-t['strain_0'][:,None], t['instant_rate_obs']]
for variant in range(4):
 metrics=[]
 for lab in range(4):
  val=t['lab_id']==lab; tau,c=m._fit(tuple(v[~val] for v in a),dt,times,y[~val],variant); r=m._design(tuple(v[val] for v in a),dt,times,tau,variant)@c-y[val]; metrics.append(np.sqrt(np.mean(r**2,axis=0))); print(variant,lab,'tau',tau.round(4),'curve',np.sqrt(np.mean(r[:,:-1]**2)),'endpoint',metrics[-1][-2],'rate',metrics[-1][-1],flush=True)
 print('max per horizon',np.max(metrics,axis=0).round(5),flush=True)
