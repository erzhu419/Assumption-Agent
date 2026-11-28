import sys, pickle, json, numpy as np
from scipy.optimize import lsq_linear
sys.path.insert(0,'/holdout_task')
import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['sensor_calibration_sweeps']; x=t['ref_rosette']; raw=t['raw_rosette']; ok=abs(raw)<4
try:
 fit=m._sweep_fit(s); print('fit',fit); print('sweep rmse',np.sqrt(np.mean((m._raw(x,fit)[ok]-raw[ok])**2))); y=m._calibrate(s); print('inverse roundtrip',np.max(abs(m._raw(y.reshape(-1,4),fit).ravel()-s['y_obs_raw'])))
except Exception as e:
 print('calibration failed',repr(e)); raise
from scipy.sparse import csc_matrix, diags, eye, kron
nx,ny=s['nx'],s['ny']; Z=s['Z_nuisance']; R=s['source_ridge']; uz=s['u_box']; budget=s['total_budget']; nz=s['nuisance_ridge']
def axis(k,v,h,rob,mode):
 if mode=='p':
  i=np.arange(len(k)); jm=(i-1)%len(k); jp=(i+1)%len(k)
  l=(k+k[jm])/(2*h*h)+np.maximum(v,0)/h; r=(k+k[jp])/(2*h*h)+np.maximum(-v,0)/h
  return csc_matrix((np.r_[l+r,-l,-r],(np.tile(i,3),np.r_[i,jm,jp])),shape=(len(k),len(k)))
 return m._axis(k,v,h,rob*float(mode))
def design(a,b):
 ax=axis(s['kappa_x'],s['vel_x'],s['Lx']/(nx-1),s['robin_x'],a); ay=axis(s['kappa_y'],s['vel_y'],s['Ly']/(ny-1),s['robin_y'],b)
 return m._responses(s,kron(eye(ny),ax,format='csc')+kron(ay,eye(nx),format='csc')+diags(s['V_reaction_2d'].ravel(order='F'),format='csc'))[0]
def evaluate(S,cv=True):
 def pred(tr,te):
  A,B=Z[tr],np.column_stack([S[tr],y[tr]]); P=np.linalg.solve(A.T@A+nz*np.eye(A.shape[1]),A.T); D=B-A@(P@B); X,Y=D[:,:-1],D[:,-1]; w,_=m._box_simplex(X.T@X+R*np.eye(X.shape[1]),-X.T@Y,uz,budget); err=S[te]@w+Z[te]@(P@(B[:,-1]-B[:,:-1]@w))-y[te]; return np.sqrt(np.mean(err**2)),w
 rows=np.arange(len(y)); e,w=pred(rows,rows); groups=np.tile(np.arange(len(y)//12),3).repeat(4); folds=[pred(rows[groups!=i],rows[groups==i])[0] for i in np.unique(groups)] if cv else [e]; return (e,max(folds),np.sqrt(np.mean(np.square(folds))),w)
for a in (0,1,'p'):
 for b in (0,1,'p'):
  e,f,q,w=evaluate(design(a,b)); print('wall',a,b,'fit/max-CV/CV',np.round([e,f,q],7),'w',np.round(w,4))
print('wall parameters',s['robin_x'],s['robin_y']);
for r in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 a=np.asarray(r['facility_telemetry_trace']); print(r['completed_campaign_short_name'],a.shape,np.round(a.mean(0),4),np.round(a[-6:].mean(0),4),np.round(a.std(0),4))
