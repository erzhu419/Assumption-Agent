import sys,json,pickle
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy.sparse import diags,kron,eye
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
from _verified_upstream import calibrate_upstream
D='/holdout_task/'
item=pickle.load(open(D+'public_batch.pkl','rb'))[0]; st=item['study']; pay=json.load(open(D+'_upstream_param_payload.json'))
Y=np.array([calibrate_upstream(dict(st,y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw'] for p in item['campaign_diagnostics']]).T
W=np.array([p['w_pulse'] for p in item['campaign_diagnostics']]).T

def axis(k,v,h,rob,p):
 n=len(k); face=(k[1:]+k[:-1])/(2*h*h)
 down=np.r_[face[:-1]+np.maximum(v[1:-1],0)/h,k[-1]/h**2]; up=np.r_[k[0]/h**2, face[1:]+np.maximum(-v[1:-1],0)/h]
 a=diags([-down,np.r_[up[0],down[:-1]+up[1:],down[-1]],-up],[-1,0,1],format='lil')
 for i,j,side in [(0,1,0),(n-1,n-2,1)]:
  robin,scale,wrap=p[side]; a[i,i]=scale*k[i]/h**2+robin*rob/h+wrap/h**2; a[i,j]=-scale*k[i]/h**2; a[i,n-1-i]=-wrap/h**2
 return a.tocsc()

def design(p):
 nx,ny=st['nx'],st['ny']; ix,iy=st['sensor_ij'].T; rhs=st['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T
 ax,ay=[axis(st['kappa_'+a],st['vel_'+a],st['L'+a]/(n-1),st['robin_'+a],np.asarray(p).reshape(2,2,3)[j]) for j,(a,n) in enumerate(zip('xy',(nx,ny)))]; I=eye(nx*ny,format='csc')
 A=(kron(eye(ny),ax)+kron(ay,eye(nx))+diags(st['V_reaction_2d'].ravel(order='F'))).tocsc(); return np.vstack([splu(A+mu*I).solve(rhs)[ix+nx*iy] for mu in st['mu_shifts']])

base=np.tile([1.,1.,0.],4); seal=np.tile([0.,1.,0.],4); z=st['Z_nuisance']; q=np.linalg.qr(z)[0]
def err(p,nuis=False):
 r=design(p)@W-Y; return r-q@(q.T@r) if nuis else r

def report(label,p):
 print(label,'per pulse raw',np.sqrt(np.mean(err(p)**2,axis=0)), 'projected',np.sqrt(np.mean(err(p,True)**2,axis=0)))
print('Y rms by pulse',np.sqrt(np.mean(Y**2,axis=0))); print('W:',W.T)
for a in (0,1):
 for b in (0,1):report(f'Robin x={a}, y={b}',np.array([[a,1,0]]*2+[[b,1,0]]*2))
for mode in ('robin','local','wrap'):
 mask=np.tile({'robin':[1,0,0],'local':[1,1,0],'wrap':[1,1,1]}[mode],4).astype(bool)
 def put(v):
  p=base.copy();p[mask]=v;return p
 for hold in (None,0,1,2):
  train=[j for j in range(3) if j!=hold]; result=least_squares(lambda v:err(put(v))[:,train].ravel(),base[mask],diff_step=1e-5,gtol=1e-13,max_nfev=200)
  p=put(result.x); print(mode,'holdout',hold,'coefficients',p.reshape(-1,3).round(7).tolist()); report('fit',p)
log=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for ent in log:
 ar=np.asarray(ent['facility_telemetry_trace']); print(ent['completed_campaign_short_name'],'mean',ar.mean(0).round(4),'std',ar.std(0).round(4))
