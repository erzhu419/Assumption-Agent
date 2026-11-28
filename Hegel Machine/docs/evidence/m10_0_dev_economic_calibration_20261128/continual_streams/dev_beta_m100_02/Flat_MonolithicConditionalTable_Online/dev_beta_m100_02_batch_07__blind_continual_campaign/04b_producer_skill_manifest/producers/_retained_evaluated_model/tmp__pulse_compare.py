import glob, importlib.util, pickle
import numpy as np
from scipy.sparse import eye,kron,diags
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; c=s['sensor_calibration_sweeps']; ref,raw=c['ref_rosette'],c['raw_rosette']; ids=c['single_bank_sweeps']['bank_id']; sx=c['single_bank_sweeps']['ref_signal']; sy=c['single_bank_sweeps']['raw_readout']; a=np.array([np.linalg.lstsq(np.c_[np.r_[ref[:,j],sx[ids==j]],np.ones(len(ref)+sum(ids==j))],np.r_[raw[:,j],sy[ids==j]],rcond=None)[0] for j in range(4)]); W=np.stack([p['w_pulse'] for p in item['campaign_diagnostics']],axis=1); Y=np.stack([((p['y_pulse_raw'].reshape(-1,4)-a[:,1])/a[:,0]).ravel() for p in item['campaign_diagnostics']],axis=1); Z=s['Z_nuisance']; print('affine coefficients',a); print('pulse RMS',np.sqrt(np.mean(Y**2,axis=0))); print('bank ids',s['row_bank_id'][:20]);
mods={}
for p in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 name=p.split('/')[-1]; spec=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); mods[name]=m
m6=next(m for name,m in mods.items() if 'batch_06' in name); m4=next(m for name,m in mods.items() if 'batch_04' in name)
def report(name,L):
 S,_=m6._responses(s,L); e=(S@W-Y); er=e-Z@np.linalg.lstsq(Z,e,rcond=None)[0]; print(name, 'raw',np.round(np.sqrt(np.mean(e**2,axis=0)),9),'projected',np.round(np.sqrt(np.mean(er**2,axis=0)),9),'per_shift',np.round(np.sqrt(np.mean(e.reshape(3,-1,3)**2,axis=(1,2))),9))
for name,m in mods.items(): report(name,m._operator(s))
def axis(d,kind):
 n=s['n'+d]; h=s['L'+d]/(n-1); k=s['kappa_'+d]; v=s['vel_'+d]; return m4._periodic_axis(k,v,h) if kind=='P' else m6._axis(k,v,h,s['robin_'+d] if kind=='R' else 0)
for x in 'RSP':
 for y in 'RSP':
  report(x+y,kron(eye(s['ny']),axis('x',x),format='csc')+kron(axis('y',y),eye(s['nx']),format='csc')+diags(s['V_reaction_2d'].ravel(order='F'),format='csc'))
