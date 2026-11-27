import sys, pickle, numpy as np
sys.path.insert(0, '/holdout_task')
import batch_solver as s
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']
tr,te,pi=[p[k] for k in ['train','test','chamber_verification_trials']]
dt,t=p['past_dt'],p['eval_times']; b=s._bridge(p['calibration_load_cells']); x=s._align([tr,te,pi],0)
for kind in ('chronological','reversed','first_extended','hold_extended','normalized'):
 for name,g,z in [('train',tr,x[0]),('pilot',pi,x[2])]:
  tau,theta,sc=s._fit(g,z,b,dt,t,kind); r=s._design(g,z,b,dt,t,tau,kind)@theta-s._obs(g)
  print(kind,name,'tau',tau.round(4),'theta',theta.round(5),'median abs',np.median(abs(r[:,:-1])).round(5),np.median(abs(r[:,-1])).round(5),'p90',np.quantile(abs(r[:,:-1]),.9).round(5),np.quantile(abs(r[:,-1]),.9).round(5),'scale',sc[[0,-1]].round(5))
