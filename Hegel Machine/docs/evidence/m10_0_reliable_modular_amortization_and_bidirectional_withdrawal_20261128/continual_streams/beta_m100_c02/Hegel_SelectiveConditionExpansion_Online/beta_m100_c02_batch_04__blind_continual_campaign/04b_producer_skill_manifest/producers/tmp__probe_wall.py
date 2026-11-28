import sys,pickle,itertools,numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
with open('/holdout_task/public_batch.pkl','rb') as f: i=pickle.load(f)[0]
s=calibrate_upstream(dict(i['study'],reference_tracer_pulses=i['campaign_diagnostics']))
p=s['reference_tracer_pulses']; W=np.stack([a['w_pulse'] for a in p],axis=1); Y=np.stack([a['y_pulse_raw'] for a in p],axis=1)
nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; idx=ij[:,0]+nx*ij[:,1]
rhs=s['source_modes'].reshape(-1,nx*ny,order='F').T@W
base=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); eye=sparse.eye(nx*ny,format='csc')
def axis(k,v,h,r,p):
 n=len(k); a=np.zeros((n,n)); faces=(k[:-1]+k[1:])/(2*h*h); j=np.arange(1,n-1)
 a[j,j-1]=-faces[:-1]-np.maximum(v[1:-1],0)/h; a[j,j+1]=-faces[1:]-np.maximum(-v[1:-1],0)/h; a[j,j]=-a.sum(1)[j]
 for end,near,far,(leak,inside,wrap) in [(0,1,n-1,p[0]),(n-1,n-2,0,p[1])]:
  a[end,near]=-inside*k[end]/h**2; a[end,far]=-wrap*(k[0]+k[-1])/(2*h*h); a[end,end]=-a[end].sum()+leak*r/h
 return sparse.csc_matrix(a)
def pred(p):
 ax,ay=[axis(s['kappa_'+d],s['vel_'+d],s['L'+d]/(s['n'+d]-1),s['robin_'+d],q) for d,q in zip('xy',np.asarray(p).reshape(2,2,3))]; op=base+sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')
 return np.concatenate([splu(op+mu*eye).solve(rhs)[idx] for mu in s['mu_shifts']])
def err(p): return pred(p)-Y
modes={'R':(1,1,0),'N':(0,1,0),'P':(0,1,1),'M':(0,2,0),'D':(0,0,0)}
r=[]
for x,y in itertools.product(modes,repeat=2):
 q=np.linalg.norm(err([modes[x]]*2+[modes[y]]*2),axis=0)/np.linalg.norm(Y,axis=0); r.append((q.max(),x+y,q.round(7)))
for v in sorted(r,key=lambda t:t[0])[:10]+[a for a in r if a[1]=='RR']: print(v)
for free in ('leak','local','full'):
 q0=np.tile([0.05,1,0.05],(4,1)); js={'leak':[0],'local':[0,1],'full':[0,1,2]}[free]; fixed=list(set(range(3))-set(js)); q0[:,fixed]=np.array([0,1,0])[fixed]
 def pars(z):
  q=q0.copy(); q[:,js]=z.reshape(4,len(js)); return q
 for held in [-1,0,1,2]:
  keep=[j for j in range(3) if j!=held]; fit=least_squares(lambda z:err(pars(z))[:,keep].ravel(),q0[:,js].ravel(),bounds=(0,np.inf),max_nfev=100,gtol=1e-13)
  print(free,held,pars(fit.x).round(6).tolist(),(np.linalg.norm(err(pars(fit.x)),axis=0)/np.linalg.norm(Y,axis=0)).round(8))
