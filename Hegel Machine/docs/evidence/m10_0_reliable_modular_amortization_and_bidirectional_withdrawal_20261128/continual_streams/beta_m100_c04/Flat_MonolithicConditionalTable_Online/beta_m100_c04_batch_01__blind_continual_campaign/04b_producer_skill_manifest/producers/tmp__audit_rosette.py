import importlib.util,pickle
import numpy as np
from scipy.optimize import least_squares
p=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(p); p.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=d['study']; c=s['sensor_calibration_sweeps']; b=c['single_bank_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; bank=b['bank_id']; print('bank ids',np.unique(bank,return_counts=True)); bank=bank-np.min(bank); br=b['ref_signal']; by=b['raw_readout']; S,_=m._forward(s)

def fit(kind,idx):
 if kind=='full':
  t=np.linalg.lstsq(np.c_[np.ones(len(idx)),x[idx]],y[idx],rcond=None)[0]
  return t[0],t[1:],np.zeros(4)
 pars=[]
 for j in range(4):
  a,z=x[idx,j],y[idx,j]; mask=bank==j
  if kind=='bank': a,z=np.r_[a,br[mask]],np.r_[z,by[mask]]
  q,g=np.linalg.lstsq(np.c_[np.ones(len(a)),a],z,rcond=None)[0]
  if kind=='rational':
   q,g,r=least_squares(lambda t:t[0]+t[1]*a/(1+t[2]*a*a)-z,[q,g,0],bounds=([-np.inf,-np.inf,-0.1],[np.inf,np.inf,0.1]),gtol=1e-13).x
  else: r=0.
  pars.append((q,g,r))
 q,g,r=np.array(pars).T; return q,np.diag(g),r

def invert(raw,q,A,r):
 z=np.linalg.solve(A.T,(raw.reshape(-1,4)-q).T).T; v=z.copy()
 for _ in range(6):
  den=1+r*v*v; v-=(v/den-z)*den**2/(1-r*v*v)
 return v.ravel()

wall=((s['sensor_ij']==0)|(s['sensor_ij']==[s['nx']-1,s['ny']-1])).any(1)
for kind in ['full','diagonal','bank','rational']:
 errs=[]
 for fold in range(5):
  test=np.arange(len(x))%5==fold; q,A,r=fit(kind,np.flatnonzero(~test)); residual=(x[test]/(1+r*x[test]**2))@A+q-y[test]; errs.append(np.sqrt(np.mean(residual**2,axis=0)))
 q,A,r=fit(kind,np.arange(len(x))); pulse=[]
 for p in d['campaign_diagnostics']:
  e=(S@p['w_pulse']-invert(p['y_pulse_raw'],q,A,r)).reshape(-1,len(wall)); pulse.extend(np.sqrt(np.mean(e[:,mask]**2,axis=1)).max() for mask in [wall,~wall])
 orig=m._calibrate; m._calibrate=lambda a:invert(a['y_obs_raw'],q,A,r); o=m.solve_batch(s); m._calibrate=orig
 ref=m.solve_batch(s)
 print(kind,'worst sweep fold/channel',max(np.ravel(errs)),'worst pulse/shift/wall slice',max(pulse),'weight relative change',np.linalg.norm(o['weights']-ref['weights'])/np.linalg.norm(ref['weights']),'field relative change',np.linalg.norm(o['terminal_field']-ref['terminal_field'])/np.linalg.norm(ref['terminal_field']),'saturation',r)
 if kind=='full': print('cross-channel terms',A-np.diag(np.diag(A)))
