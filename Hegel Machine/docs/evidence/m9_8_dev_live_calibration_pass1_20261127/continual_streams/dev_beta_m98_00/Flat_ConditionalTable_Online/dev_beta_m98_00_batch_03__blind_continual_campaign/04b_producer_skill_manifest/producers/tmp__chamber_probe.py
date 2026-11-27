import importlib.util,pickle,numpy as np,json
from scipy.optimize import least_squares
path='/tmp/hegel_continual_archive/'
spec=importlib.util.spec_from_file_location('arc',path+'dev_beta_m98_00_batch_02_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; src=s['source_modes']; cal=m._calibration(s['sensor_calibration_sweeps']); ps=s['reference_tracer_pulses']; modes=np.array([np.tensordot(p['w_pulse'],src,axes=1) for p in ps]); obs=np.column_stack([cal(p['y_pulse_raw']) for p in ps]); ij=s['sensor_ij']; print('shifts',s['mu_shifts'],'V quantiles',np.quantile(s['V_reaction_2d'],[0,.25,.5,.75,1]),'pulse weights',np.array([p['w_pulse'] for p in ps])); print('sensors',ij.T); print('avg observed',obs.reshape(3,-1,2).mean(1)); print('telemetry',np.round(s['aux_manifold_telemetry'][:8],2));
for e in json.load(open(path+'archive_apparatus_log.json')):
 print('log',[(k,np.round(np.array(v).mean(0),3).tolist() if 'trace' in k else v) for k,v in e.items()]);
def predict(q,kind):
 st=s.copy(); w=np.ones(8)
 if kind=='signed_walls': w=q
 if kind=='seal': w[[0,1,4,5]]=0
 if kind=='lined': w=q[:8]; st['V_reaction_2d']=s['V_reaction_2d']+q[8]; st['mu_shifts']=s['mu_shifts']*np.exp(q[9]);
 if kind=='bulk':
  w=q[:8]; st['V_reaction_2d']=s['V_reaction_2d']+q[8]; st['mu_shifts']=s['mu_shifts']*np.exp(q[9]);
  for k,i in [('kappa_x',10),('kappa_y',11),('vel_x',12),('vel_y',13)]: st[k]=s[k]*np.exp(q[i])
 return m._response(st,w,modes)[0]
for k,x in [('seal',[]),('signed_walls',np.ones(8)),('lined',np.r_[np.ones(8),0,0]),('bulk',np.r_[np.ones(8),np.zeros(6)])]:
 if not len(x):
  r=predict(x,k)-obs; print(k,np.sqrt((r.reshape(3,-1,2)**2).mean(1)));continue
 lower=np.r_[np.full(2,-1.),[0.01]*2,np.full(2,-1.),[0.01]*2,[-.4,-2.,-2.,-2.,-2.,-2.]][:len(x)]; upper=np.r_[np.full(8,20.),[2.,2.,2.,2.,2.,2.]][:len(x)];
 for j in [-1,0,1]:
  keep=slice(None) if j<0 else j; fit=least_squares(lambda q:(predict(q,k)-obs)[:,keep].ravel(),x,bounds=(lower,upper),max_nfev=100); r=predict(fit.x,k)-obs
  print(k,j,np.round(fit.x,5),np.round(np.sqrt((r.reshape(3,-1,2)**2).mean(1)),7).tolist());
