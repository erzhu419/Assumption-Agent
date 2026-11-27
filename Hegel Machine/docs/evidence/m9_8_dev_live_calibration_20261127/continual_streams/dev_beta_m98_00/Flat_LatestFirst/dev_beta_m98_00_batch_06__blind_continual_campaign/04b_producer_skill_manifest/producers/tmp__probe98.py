import importlib.util, pickle, numpy as np
from pathlib import Path
from scipy.linalg import lstsq
from scipy.optimize import root
np.set_printoptions(precision=5,suppress=True)
e=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=e['study']; c=s['sensor_calibration_sweeps']; X,Y=c['ref_rosette'],c['raw_rosette']; Z=s['Z_nuisance']
mods={}
for f in sorted(Path('/tmp/hegel_continual_archive').glob('*_solver.py')):
 name=f.stem.removeprefix('dev_beta_m98_00_').removesuffix('_solver'); spec=importlib.util.spec_from_file_location(name,f); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);mods[name]=m
print('sweep range',X.min(0),X.max(0),'raw range',Y.min(0),Y.max(0)); print('bank sweep fields',[(k,type(v).__name__,getattr(v,'shape',None),list(v) if isinstance(v,dict) else '') for k,v in c['single_bank_sweeps'].items()]);print('sensor extrema',s['sensor_ij'].min(0),s['sensor_ij'].max(0),'mu',s['mu_shifts']); print('Z/raw norm',np.linalg.norm(Z,axis=0),np.linalg.norm(s['y_obs_raw']))
terms={0:lambda x:np.c_[np.ones(len(x)),x],1:mods['batch_04']._rosette_terms,2:lambda x:np.c_[np.ones(len(x)),x,*[(x[:,i]*x[:,j]) for i in range(4) for j in range(i,4)],*[(x[:,i]*x[:,j]*x[:,k]) for i in range(4) for j in range(i,4) for k in range(j,4)]]}
ys={}
for deg,fn in terms.items():
 F=fn(X); A=[]; rr=[]
 for idx in np.array_split(np.arange(len(X)),5)+[np.array([],dtype=int)]:
  train=np.ones(len(X),bool);train[idx]=False; keep=train & (np.max(abs(Y),axis=1)<4*np.median(np.max(abs(Y[train]),axis=1)))
  for _ in range(5):
   fit=np.linalg.lstsq(F[keep],Y[keep],rcond=None)[0];r=np.max(abs(F@fit-Y),axis=1); med=np.median(r[train]); keep=train & (r<=med+8*max(np.median(abs(r[train]-med)),1e-12))
  if len(idx):rr.extend(r[idx]);continue
  print('cal deg',deg,'heldout abs quantiles',np.quantile(rr,[.5,.8,.9,1]),'kept',keep.sum(),'training median',np.median(r[keep]));
  if deg<=1: print('fit',fit)
  def invert(raw):
   raw=raw.reshape(-1,4); x0=np.linalg.solve(fit[1:5].T,(raw-fit[0]).T).T
   out=[root(lambda x:(fn(x[None,:])@fit)[0]-y,x).x for x,y in zip(x0,raw)]; return np.ravel(out)
  ys[str(deg)]=[invert(p['y_pulse_raw']) for p in e['campaign_diagnostics']]
ys['archive04']=[mods['batch_04']._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in e['campaign_diagnostics']]
for name,m in mods.items():
 S,_=m._design(s)
 for cal,ylist in ys.items():
  vals=[]
  for p,y in zip(e['campaign_diagnostics'],ylist):
   d=y-S@p['w_pulse']; zn=Z@lstsq(Z,d)[0]; vals.append((round(np.linalg.norm(d),6),round(np.linalg.norm(d-zn),6)))
  print('pulse raw/projected norms',name,cal,vals)
