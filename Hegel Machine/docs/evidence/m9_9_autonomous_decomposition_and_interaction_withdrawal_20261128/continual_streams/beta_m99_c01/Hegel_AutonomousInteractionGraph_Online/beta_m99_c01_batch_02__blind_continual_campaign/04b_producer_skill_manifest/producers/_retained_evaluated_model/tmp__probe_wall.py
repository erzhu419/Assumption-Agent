import sys, pickle, json
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy.sparse import diags, eye, kron, csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
from _verified_upstream import calibrate_upstream
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; st=item['study']; pay=json.load(open('/holdout_task/_upstream_param_payload.json'))
nx,ny=st['nx'],st['ny']; ix,iy=st['sensor_ij'].T
modes=st['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T
W=np.array([p['w_pulse'] for p in item['campaign_diagnostics']]); Y=np.array([calibrate_upstream({'y_obs_raw':p['y_pulse_raw']},pay)['y_obs_raw'] for p in item['campaign_diagnostics']]); print('pulse range',Y.min(),Y.max(),'weights',W)
I=eye(nx*ny,format='csc'); V=diags(st['V_reaction_2d'].ravel(order='F'),format='csc')
def axis(key,p):
 k,v=st['kappa_'+key],st['vel_'+key]; n=len(k); h=st['L'+key]/(n-1); r=st['robin_'+key]/h
 f=(k[:-1]+k[1:])/(2*h*h); a,b=-f[:-1]-np.maximum(v[1:-1],0)/h,-f[1:]-np.maximum(-v[1:-1],0)/h
 l0,l1,d0,d1,c0,c1=p; q0=d0*k[0]/h**2; q1=d1*k[-1]/h**2
 A=diags((np.r_[a,-q1],np.r_[q0+l0*r+c0,-a-b,q1+l1*r+c1],np.r_[-q0,b]),(-1,0,1),format='csc')
 return A+csc_matrix(([-c0,-c1],([0,n-1],[n-1,0])),shape=(n,n))
def predict(p):
 A=kron(eye(ny),axis('x',p[:6]),format='csc')+kron(axis('y',p[6:]),eye(nx),format='csc')+V
 return np.concatenate([splu(A+mu*I).solve(modes@W.T)[ix+nx*iy].T for mu in st['mu_shifts']],axis=1)
def report(p):
 e=predict(p)-Y; print('rms per pulse',np.sqrt(np.mean(e*e,axis=1)),'maxabs',np.max(abs(e)))
base=np.tile([1.,1.,1.,1.,0.,0.],2); refl=base.copy(); refl[[0,1,6,7]]=0
print('Robin'); report(base); print('reflecting'); report(refl)
for free in ([0,1,6,7],[0,1,2,3,6,7,8,9],list(range(12))):
 p=base.copy()
 for omit in [-1,0,1,2]:
  keep=np.arange(len(W))!=omit
  def unpack(q):
   p=p0.copy(); p[free]=q; return p
  p0=base.copy(); fit=least_squares(lambda q:(predict(unpack(q))-Y)[keep].ravel(),p0[free],gtol=1e-13,xtol=1e-12,ftol=1e-12,max_nfev=180)
  p=unpack(fit.x); print('free',len(free),'omit',omit,'pars',p.round(7)); report(p)
a=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for entry in a:
 t=np.array(entry['facility_telemetry_trace']); print(entry['completed_campaign_short_name'],'mean',t.mean(0).round(5),'std',t.std(0).round(5),'row0',t[0])
print('sweep fields',list(st['sensor_calibration_sweeps'])); x=st['sensor_calibration_sweeps']['ref_rosette']; raw=st['sensor_calibration_sweeps']['raw_rosette']; m=np.max(abs(raw),axis=1)<=8.5; xx=np.c_[x[m],np.ones(m.sum())]; yy=raw[m]; beta=np.linalg.lstsq(xx,yy,rcond=None)[0]; print('affine sweep fits',beta, 'max residual',np.max(abs(xx@beta-yy),axis=0))
