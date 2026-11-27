import sys,pickle,json,itertools
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy.sparse import csc_matrix,diags,eye,kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
from _verified_upstream import calibrate_upstream
p=pickle.load(open('public_batch.pkl','rb'))[0]; st=p['study']; s=calibrate_upstream(dict(st,reference_tracer_pulses=p['campaign_diagnostics']))
pulses=s['reference_tracer_pulses']; W=np.array([p['w_pulse'] for p in pulses]).T; Y=np.array([p['y_pulse_raw'] for p in pulses]).T
print('pulse weights',W.T,'norms',np.linalg.norm(Y,axis=0),flush=True)
for a in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 t=np.array(a['facility_telemetry_trace']); print(a['completed_campaign_short_name'], t.mean(0).round(3))
print('current',st['aux_manifold_telemetry'].mean(0).round(3),flush=True)
nx,ny=s['nx'],s['ny']; rhs=s['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T@W; ij=s['sensor_ij']; pos=ij[:,0]+nx*ij[:,1]; eye2=eye(nx*ny,format='csc'); V=diags(s['V_reaction_2d'].ravel(order='F'))
def axis(k, kind):
 kap=s['kappa_'+k]; v=s['vel_'+k]; rob=s['robin_'+k]; n=len(kap); h=s['L'+k]/(n-1); A=np.zeros((n,n)); i=np.arange(1,n-1); km=(kap[i]+kap[i-1])/(2*h*h)+np.maximum(v[i],0)/h; kp=(kap[i]+kap[i+1])/(2*h*h)+np.maximum(-v[i],0)/h; A[i,i-1]=-km; A[i,i+1]=-kp; A[i,i]=km+kp
 for j,(i,m,z) in enumerate(((0,1,n-1),(n-1,n-2,0))):
  b=kap[i]/h**2; r=rob/h; q=(kap[i]+kap[m])/(2*h*h)+max((-1 if j==0 else 1)*v[i],0)/h; seam=(kap[i]+kap[z])/(2*h*h)+max((1 if j==0 else -1)*v[i],0)/h
  if isinstance(kind,str):
   a,e,l={'open':(b,0,r),'sealed':(b,0,0),'periodic':(q,seam,0),'local_periodic':(b,b,0),'upwind_open':(q,0,r),'upwind_sealed':(q,0,0)}[kind]; second=0
  else:
   a,e,l,second=kind.reshape(2,4)[j]*[b,b,r,b]
  A[i,m]=-a; A[i,z]=-e; A[i,i]=a+e+l+second; A[i,2 if j==0 else n-3]=-second
 return csc_matrix(A)
def pred(x,y):
 L=(kron(eye(ny),axis('x',x))+kron(axis('y',y),eye(nx))+V).tocsc()
 return np.vstack([splu(L+float(mu)*eye2).solve(rhs)[pos] for mu in s['mu_shifts']])
def summary(e):
 return np.sqrt(np.mean(e**2,axis=0)).round(9),float(np.max(abs(e)))
if __name__=='__main__':
 kinds=['open','sealed','periodic','local_periodic','upwind_open','upwind_sealed']; a=[]
 for x,y in itertools.product(kinds,repeat=2):
  e=pred(x,y)-Y; a.append((np.linalg.norm(e),x,y,summary(e)))
 print('discrete',sorted(a)[:10],flush=True)
 center=np.tile([1.,0.,1.,0.],4); models={}
 for label,cols in [('neighbor/leak',[0,2,4,6,8,10,12,14]),('neighbor/leak/seam',[0,1,2,4,5,6,8,9,10,12,13,14]),('all',list(range(16)))]:
  def expand(z):
   p=center.copy(); p[cols]=z; return p
  def res(z):
   p=expand(z); return pred(p[:8],p[8:])-Y
  z=least_squares(lambda z:res(z).ravel(),center[cols],gtol=1e-13,ftol=1e-13,xtol=1e-13,max_nfev=100)
  print(label,expand(z.x).reshape(4,4).round(8),summary(res(z.x)),np.linalg.svd(z.jac,compute_uv=False).round(7),flush=True)
  for held in range(W.shape[1]):
   zz=least_squares(lambda w:np.delete(res(w),held,axis=1).ravel(),center[cols],gtol=1e-11,max_nfev=80)
   print('held',held,summary(res(zz.x)[:,held]),expand(zz.x).reshape(4,4).round(5),flush=True)
