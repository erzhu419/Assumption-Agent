import importlib.util,itertools,pickle,numpy as np
from pathlib import Path
p=Path('/tmp/hegel_continual_archive')
def load(path):
 spec=importlib.util.spec_from_file_location('candidate',path); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
a=load(p/'beta_m100_c01_batch_05_solver.py'); c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; keep=np.max(abs(y),axis=1)<5
for path in sorted(p.glob('*solver.py')):
 print(path.name); text=path.read_text().splitlines(); print('\n'.join(line for line in text if ('_transport(' in line or 'def _response' in line or 'def _design' in line)))
for n in ('03','04','05'):
 m=load(p/f'beta_m100_c01_batch_{n}_solver.py'); errs=[]
 for k in range(5):
  test=(np.arange(len(x))%5==k)&keep; t=(np.arange(len(x))%5!=k)
  if n=='05':
   b=c['single_bank_sweeps']; b={j:v[np.arange(len(v))%5!=k] for j,v in b.items()}; q=m._fit_calibration(x[t],y[t],b); pred=m._response(x[test],q)
  else:
   q=m._fit_calibration(x[t],y[t]); pred=m._forward(x[test],q)
  errs.append(np.sqrt(np.mean((pred-y[test])**2)))
 print('sweep CV',n,errs); print('parameters',q)
y=a._calibrate(s); Z=s['Z_nuisance']; zreg=s['nuisance_ridge']; lam=s['source_ridge']; u=s['u_box']; B=s['total_budget']; ZI=np.eye(Z.shape[1])*zreg
for ww in itertools.product('RNP',repeat=2):
 S,U=a._transport(s,ww); scores=[]; foldpred=np.zeros(len(y)); coords=np.tile(s['sensor_ij'],(len(s['mu_shifts']),1))
 for j,ij in enumerate(np.unique(coords,axis=0)):
  test=np.all(coords==ij,axis=1); t=~test; z=Z[t]; X=np.column_stack([S[t],y[t]]); projected=X-z@np.linalg.solve(z.T@z+ZI,z.T@X); A,r=projected[:,:-1],projected[:,-1]; w,nu=a._box_simplex(A.T@A+lam*np.eye(A.shape[1]),-A.T@r,u,B); gam=np.linalg.solve(z.T@z+ZI,z.T@(y[t]-S[t]@w)); foldpred[test]=S[test]@w+Z[test]@gam; scores.append(np.sqrt(np.mean((foldpred[test]-y[test])**2)))
 X=np.column_stack([S,y]); projected=X-Z@np.linalg.solve(Z.T@Z+ZI,Z.T@X); A,r=projected[:,:-1],projected[:,-1]; w,nu=a._box_simplex(A.T@A+lam*np.eye(A.shape[1]),-A.T@r,u,B); err=A@w-r
 print('wall',''.join(ww),'CV RMS/worst',np.sqrt(np.mean((foldpred-y)**2)),max(scores),'in-sample RMS/max',np.sqrt(np.mean(err**2)),max(abs(err)),'w',np.round(w,4))
