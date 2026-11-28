import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
from scipy.sparse import eye,kron,diags
p='/tmp/hegel_continual_archive/dev_beta_m100_02_batch_03_solver.py'
a=importlib.util.spec_from_file_location('prior',p); m=importlib.util.module_from_spec(a); a.loader.exec_module(m)
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=it['study']; ds=it['campaign_diagnostics']; c=s['sensor_calibration_sweeps']; t=m._sweep_fit(s)
print('coeffs',np.round(t,7)); e=m._raw(c['ref_rosette'],t)-c['raw_rosette']; print('rosette absolute residual quantiles',np.quantile(abs(e),[.5,.9,.95,.98,1],axis=0));
Y=np.stack([m._calibrate(dict(s,y_obs_raw=d['y_pulse_raw'])) for d in ds],axis=1); W=np.stack([d['w_pulse'] for d in ds],axis=1)
nx,ny=s['nx'],s['ny']; Z=s['Z_nuisance']; rz=Z@np.linalg.pinv(Z); hx=s['Lx']/(nx-1); hy=s['Ly']/(ny-1)
ax=m._axis(s['kappa_x'],s['vel_x'],hx,0); ay=m._axis(s['kappa_y'],s['vel_y'],hy,0)
base=kron(eye(ny),ax,format='csc')+kron(ay,eye(nx),format='csc')+diags(s['V_reaction_2d'].ravel(order='F'),format='csc')

def design(p):
    leak,diff=np.asarray(p).reshape(2,4); axes=[]
    for q,(n,h,k,r) in enumerate([(nx,hx,s['kappa_x'],s['robin_x']),(ny,hy,s['kappa_y'],s['robin_y'])]):
        b=np.zeros((n,n)); z=2*q
        for j,i,i2 in [(z,0,1),(z+1,n-1,n-2)]:
            d=diff[j]*k[i]/h**2; b[i,i]=d+leak[j]*r/h; b[i,i2]=-d
        from scipy.sparse import csc_matrix
        axes.append(csc_matrix(b))
    return m._responses(s,base+kron(eye(ny),axes[0],format='csc')+kron(axes[1],eye(nx),format='csc'))[0]@W

def err(p,project=False):
    e=design(p)-Y
    return e-rz@e if project else e
import json
from scipy.sparse import csc_matrix
print('W',W.T)
for i,d in enumerate(ds): print('pulse keys',i,list(d))
e=err(np.zeros(8)).reshape(3,-1,3); ij=s['sensor_ij']; print('shift mean',e.mean(axis=1),'shift RMS',np.sqrt((e**2).mean(axis=1)))
for j in range(len(ij)): print(ij[j],np.round(e[:,j,:].T,4).tolist())
for r in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 a=np.asarray(r['facility_telemetry_trace']); print(r['completed_campaign_short_name'],'means',np.round(a.mean(axis=0),3),'sd',np.round(a.std(axis=0),3))
print('study telemetry mean/std',s['aux_manifold_telemetry'].mean(axis=0),s['aux_manifold_telemetry'].std(axis=0))
def operator(p):
    # Relative endpoint leak, incident diffusion, and wrap conductance.
    leak,diff,wrap=np.asarray(p).reshape(3,4); axes=[]
    for q,(n,h,k,r) in enumerate([(nx,hx,s['kappa_x'],s['robin_x']),(ny,hy,s['kappa_y'],s['robin_y'])]):
        b=np.zeros((n,n)); z=2*q
        for j,i,i2,opp in [(z,0,1,n-1),(z+1,n-1,n-2,0)]:
            d=diff[j]*k[i]/h**2; wp=wrap[j]*k[i]/h**2; b[i,i]+=d+wp+leak[j]*r/h; b[i,i2]-=d; b[i,opp]-=wp
        axes.append(csc_matrix(b))
    return base+kron(eye(ny),axes[0],format='csc')+kron(axes[1],eye(nx),format='csc')
def res(p): return m._responses(s,operator(p))[0]@W-Y
for pairing in [True,False]:
 for wrap in [False,True]:
  n=2 if pairing else 4; blocks=3 if wrap else 2
  def expand(q):
   q=np.asarray(q).reshape(blocks,n); q=np.repeat(q,2,axis=1) if pairing else q
   return (q if wrap else np.vstack((q,np.zeros(4)))).ravel()
  lb=np.r_[np.zeros(n),np.full(n,-.999),np.zeros(n) if wrap else []]; ub=np.full(n*blocks,30.)
  for hold in [-1,0,1,2]:
   use=np.arange(3)!=hold
   fit=least_squares(lambda q:res(expand(q))[:,use].ravel(),np.full(n*blocks,.03),bounds=(lb,ub),max_nfev=100)
   p=expand(fit.x); print('wall fit',pairing,wrap,hold,'params',np.round(p,4),'rms',np.round(np.sqrt(np.mean(res(p)**2,axis=0)),7))
