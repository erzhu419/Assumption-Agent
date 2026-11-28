import sys,json,pickle,itertools
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
with open('/holdout_task/public_batch.pkl','rb') as f: item=pickle.load(f)[0]
with open('/holdout_task/_upstream_param_payload.json') as f: pay=json.load(f)
s=item['study']; nx,ny=s['nx'],s['ny']; Z=s['Z_nuisance']; P=np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
pulses=item['campaign_diagnostics']; raw=np.column_stack([p['y_pulse_raw'] for p in pulses]); Y=np.column_stack([calibrate_upstream(dict(s,y_obs_raw=a),pay)['y_obs_raw'] for a in raw.T]); W=np.column_stack([p['w_pulse'] for p in pulses]); rhs=np.column_stack([m.ravel(order='F') for m in (W.T@s['source_modes'].reshape(len(W),-1)).reshape(-1,nx,ny)])
obs=s['sensor_ij']@np.array([1,nx]); eye=sparse.eye(nx*ny,format='csc'); vmat=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')

def axis(d,ends):
 k=s['kappa_'+d]; v=s['vel_'+d]; n=len(k); h=s['L'+d]/(n-1); rob=s['robin_'+d]/h
 face=(k[:-1]+k[1:])/(2*h*h); sev=(k[0]+k[-1])/(2*h*h); a=np.zeros((n,n)); ix=np.arange(1,n-1)
 a[ix,ix-1]=-face[:-1]-np.maximum(v[1:-1],0)/h;a[ix,ix+1]=-face[1:]-np.maximum(-v[1:-1],0)/h
 for idx,near,far,sgn,fa,name in [(0,1,n-1,1,face[0],ends[0]),(n-1,n-2,0,-1,face[-1],ends[1])]:
  inward=max(-sgn*v[idx],0)/h; outward=max(sgn*v[idx],0)/h
  c=k[idx]/h**2; w=0.; leak=0.
  if name=='R': leak=rob
  if name=='F': c=fa+inward
  if name=='W': c=fa+inward; w=sev+outward
  if name=='D': c=fa; w=sev
  if name=='H': w=sev
  if name=='Q': c*=2
  if name=='L': c=fa+inward; leak=rob
  a[idx,near]-=c; a[idx,far]-=w; a[idx,idx]+=leak
 a[np.diag_indices(n)]-=a.sum(axis=1)-np.diag(a)
 return sparse.csc_matrix(a)

def predict(ax,ay, b=rhs):
 op=sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')+vmat
 return np.concatenate([splu(op+mu*eye).solve(b)[obs] for mu in s['mu_shifts']])

def summarize(err):
 return np.sqrt(np.mean(err**2,axis=0))
if __name__=='__main__':
 options=list(itertools.product('RCFWDHQL',repeat=2)); axs={d:{''.join(e):axis(d,e) for e in options} for d in 'xy'}; scores=[]
 print('calibrated pulse rms',summarize(Y),'nuisance-fit pulse rms',summarize(Z@(P@Y)))
 for x,ax in axs['x'].items():
  for y,ay in axs['y'].items():
   err=predict(ax,ay)-Y; rs=summarize(err); ps=summarize(err-Z@(P@err)); scores.append((max(rs),max(ps),x,y,rs,ps))
 for row in sorted(scores,key=lambda a:a[0])[:12]: print('raw ranked',row)
 for pair in [('RR','RR'),('CC','CC'),('WW','CC'),('CC','WW'),('WW','WW')]:
  row=next(a for a in scores if a[2:4]==pair); print('baseline',row)
 for row in sorted(scores,key=lambda a:a[1])[:4]: print('nuisance-profiled ranked',row)
