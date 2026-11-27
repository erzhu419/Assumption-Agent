import importlib.util, pickle, numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; g=d['train']; x,_=m._features(g,d['test']); st=m._stress_map(d['calibration_load_cells']); dt=d['past_dt']; t=d['eval_times']; y=g['strain_curves_obs']-g['strain_0'][:,None]; v=g['instant_rate_obs']; labs=g['lab_id']; early=t<=np.median(t)
def fit(a,kind):
 ta,th=m._fit({k:z[a] for k,z in g.items()},x[a],st,dt,t)
 if kind=='current': return ta,th
 c,r=m._design(g,x,st,dt,t,ta); ey=(c@th-y)[a]; ev=(r@th-v)[a]; sy,sr=m._mad(ey),m._mad(ev); keep=np.abs(ey-np.median(ey))<4*sy
 def resid(p):
  C,R=m._design(g,x,st,dt,t,np.exp(p[:2])); dc=(C@p[2:]-y)[a]/sy; dr=(R@p[2:]-v)[a]/sr
  return np.r_[dc[keep] if kind=='trimmed' else dc.ravel(),dr]
 q=least_squares(resid,np.r_[np.log(ta),th],loss='linear' if kind=='trimmed' else 'cauchy',f_scale=2.5,bounds=(np.r_[np.log([.05000001]*2),[-np.inf]*7],np.r_[np.log([24.999999]*2),[np.inf]*7])).x
 return np.exp(q[:2]),q[2:]
for kind in ('current','trimmed','cauchy'):
 ta,th=fit(np.ones(len(x),bool),kind); scores=[]; folds=[]
 for l in np.unique(labs):
  b=labs==l; ta1,th1=fit(~b,kind); folds.append(ta1); C,R=m._design(g,x,st,dt,t,ta1); e=np.abs((C@th1-y)[b]); f=(R@th1-v)[b]; scores.append([np.median(e[:,early]),np.median(e[:,~early]),np.sqrt(np.mean(f**2))]);
 print(kind,'full taus',ta.round(6),'fold taus',np.round(folds,5).tolist(),'worst lab early/late median absolute strain error and rate RMSE',np.max(scores,0).round(6),'mean',np.mean(scores,0).round(6),'theta',th.round(5),flush=True)
