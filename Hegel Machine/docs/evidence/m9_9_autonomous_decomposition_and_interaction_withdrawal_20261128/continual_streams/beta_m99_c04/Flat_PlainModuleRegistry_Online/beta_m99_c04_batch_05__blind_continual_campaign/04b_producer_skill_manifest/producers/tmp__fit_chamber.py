import sys,pickle,numpy as np
from scipy.sparse import csc_matrix,diags,eye,kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s=calibrate_upstream(dict(p['study'],reference_tracer_pulses=p['campaign_diagnostics']))
Y=np.stack([x['y_pulse_raw'] for x in s['reference_tracer_pulses']],axis=1)
W=np.stack([x['w_pulse'] for x in s['reference_tracer_pulses']],axis=1)
nx,ny=s['nx'],s['ny']; I=eye(nx*ny,format='csc'); V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); ij=s['sensor_ij']@np.array([1,nx]); F=np.stack([a.ravel(order='F') for a in s['source_modes']],axis=1)@W
print('pulse weights',W.T)
print('study telemetry mean/std',s['aux_manifold_telemetry'].mean(0),s['aux_manifold_telemetry'].std(0))

def axis(d, pars):
 k=s['kappa_'+d]; v=s['vel_'+d]; n=len(k); h=s['L'+d]/(n-1); a=np.zeros((n,n)); i=np.arange(1,n-1); f=(k[:-1]+k[1:])/(2*h*h)
 a[i,i-1]=-f[:-1]-np.maximum(v[1:-1],0)/h; a[i,i+1]=-f[1:]-np.maximum(-v[1:-1],0)/h; a[i,i]=-a[i,i-1]-a[i,i+1]
 for side,(j,l,q) in enumerate([(0,1,n-1),(n-1,n-2,0)]):
  near,wrap,sink=pars[3*side:3*side+3]; a[j,l]=-near; a[j,q]=-wrap; a[j,j]=near+wrap+sink
 return csc_matrix(a)

def predict(t):
 L=kron(eye(ny),axis('x',t[:6]),format='csc')+kron(axis('y',t[6:]),eye(nx),format='csc')+V
 return np.concatenate([splu(L+mu*I).solve(F)[ij] for mu in s['mu_shifts']])

def base(kind):
 out=[]
 for d in 'xy':
  k,v=s['kappa_'+d],s['vel_'+d]; h=s['L'+d]/(len(k)-1); wrap=(k[0]+k[-1])/(2*h*h)
  for j,l,sign in [(0,1,-1),(-1,-2,1)]:
   if kind=='periodic': out.extend([(k[j]+k[l])/(2*h*h)+max(sign*v[j],0)/h,wrap+max(-sign*v[j],0)/h,0])
   else: out.extend([k[j]/h**2,0,s['robin_'+d]/h if kind=='robin' else 0])
 return np.array(out)

def fit(mask,kind,full):
 b=base(kind); use=np.ones(12,dtype=bool) if full else np.arange(12)%3!=2
 def unpack(x):
  t=b.copy(); t[use]=x; return t
 lo=np.tile([0,0,-0.1],4)[use]; hi=np.full(sum(use),160.)
 ans=least_squares(lambda x:(predict(unpack(x))-Y)[mask],b[use],bounds=(lo,hi),ftol=1e-10,gtol=1e-10,xtol=1e-10,max_nfev=180)
 return unpack(ans.x),np.linalg.svd(ans.jac,compute_uv=False)
for kind in ['reflecting','periodic']:
 for full in [False,True]:
  mask=np.full(Y.shape,True); t,sv=fit(mask,kind,full); err=predict(t)-Y; print(kind,full,'t',np.round(t.reshape(4,3),5),'sv',np.round(sv,5),'rmse',np.sqrt(np.mean(err**2,axis=0)))
  for pul in range(Y.shape[1]):
   mask[:,pul]=False; h,_=fit(mask,kind,full); mask[:,pul]=True; e=(predict(h)-Y)[:,pul].reshape(len(s['mu_shifts']),-1); print('holdout',pul,'shift RMSE',np.sqrt(np.mean(e**2,axis=1)))
  if kind=='periodic' and full:
   for sh,idx in enumerate(np.array_split(np.arange(len(Y)),len(s['mu_shifts']))):
    mask[idx,:]=False; h,_=fit(mask,kind,full); mask[idx,:]=True; e=(predict(h)-Y)[idx,:]; print('holdout shift',sh,'pulse RMSE',np.sqrt(np.mean(e**2,axis=0)))
np.set_printoptions(precision=4,suppress=True)
print('baseline periodic endpoints',base('periodic').reshape(4,3));print('baseline reflection endpoints',base('reflecting').reshape(4,3));print('periodic residual first shift'); print((predict(base('periodic'))-Y)[:len(ij)])
