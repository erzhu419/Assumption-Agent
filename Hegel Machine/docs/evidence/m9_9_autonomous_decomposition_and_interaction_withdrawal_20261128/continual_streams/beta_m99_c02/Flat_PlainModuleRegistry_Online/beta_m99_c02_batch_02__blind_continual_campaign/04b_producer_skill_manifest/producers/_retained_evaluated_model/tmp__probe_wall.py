import sys,json,pickle,importlib.util
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csc_matrix
from _verified_upstream import calibrate_upstream
spec=importlib.util.spec_from_file_location('ref','/tmp/hegel_continual_archive/beta_m99_c02_batch_01_solver.py'); ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; pay=json.load(open('/holdout_task/_upstream_param_payload.json')); pulses=item['campaign_diagnostics']; W=np.stack([p['w_pulse'] for p in pulses]); Y=np.stack([calibrate_upstream(dict(s,y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw'] for p in pulses]); hs=[s['L'+a]/(s['n'+a]-1) for a in 'xy']; K=[s['kappa_'+a] for a in 'xy']; V=[s['vel_'+a] for a in 'xy']; A=[ref._axis(k,v,h).toarray() for k,v,h in zip(K,V,hs)]; B=np.array([[k[0]/h**2,k[-1]/h**2] for k,h in zip(K,hs)]).ravel(); F=np.array([[(k[0]+k[1])/(2*h*h),(k[-1]+k[-2])/(2*h*h)] for k,h in zip(K,hs)]).ravel(); C=np.array([[(k[0]+k[-1])/(2*h*h)]*2 for k,h in zip(K,hs)]).ravel(); print('base,faces,wrap',B,F,C)
def predict(params):
 axes=[]
 for j,base in enumerate(A):
  a=base.copy()
  for q,(i,k,w) in enumerate([(0,1,-1),(-1,-2,0)]):
   inward,wrap,leak=params[2*j+q]; a[i,k]=-inward;a[i,w]=-wrap;a[i,i]=inward+wrap+leak
  axes.append(csc_matrix(a))
 return W@ref._responses(s,axes)[0].T

def report(name,P):
 R=(predict(P)-Y).reshape(3,3,-1); print(name,'worst slice rms',np.sqrt(np.mean(R*R,axis=-1)).max(),'global',np.sqrt(np.mean(R*R)), 'max',abs(R).max())
zero=np.zeros(4)
for label,inn in [('local',B),('face',F),('double local',2*B),('double face',2*F)]:
 for j,wrap in enumerate([zero,B,C,2*C]):
  report(label+' wrap'+str(j),np.c_[inn,wrap,zero]);
adv_in=np.array([max(-V[0][0],0)/hs[0],max(V[0][-1],0)/hs[0],max(-V[1][0],0)/hs[1],max(V[1][-1],0)/hs[1]]); adv_wrap=np.array([max(V[0][0],0)/hs[0],max(-V[0][-1],0)/hs[0],max(V[1][0],0)/hs[1],max(-V[1][-1],0)/hs[1]]);print('adv in and wrap:',adv_in,adv_wrap); report('full periodic upwind',np.c_[F+adv_in,C+adv_wrap,zero]);
for tr in ([0,1,2],[1,2],[0,2],[0,1]):
 r=least_squares(lambda p:(predict(p.reshape(4,3))-Y)[tr].ravel(),np.c_[F,C,zero].ravel(),ftol=1e-13,xtol=1e-13,gtol=1e-13,max_nfev=200); print('fit training',tr,r.x.reshape(4,3),'jac singular values',np.linalg.svd(r.jac,compute_uv=False)[[0,-1]]); report('fitted all slices',r.x.reshape(4,3))
print('telemetry summary')
for e in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 a=np.asarray(e['facility_telemetry_trace']);print(e['completed_campaign_short_name'],a.mean(0).round(3),a.std(0).round(3))
