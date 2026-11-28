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
k,v=s['kappa_x'],s['vel_x']; h=s['Lx']/(nx-1); seam=(k[0]+k[-1])/2
local=[(k[0]+k[1])/2,(k[-1]+k[-2])/2]
for avg,adv in itertools.product((False,True),repeat=2):
 p=[]
 for j,e in enumerate((0,-1)):
  inward=((-1)**(j+1))*v[e]
  p.append([0,((local[j] if avg else k[e])+adv*h*max(inward,0))/k[e],(seam+adv*h*max(-inward,0))/seam])
 p += [[0,1,0]]*2
 print('averaged faces',avg,'upwind at seam',adv,'ratios',np.array(p)[:2,1:].round(7),'relative pulse error',np.linalg.norm(err(p),axis=0)/np.linalg.norm(Y,axis=0))
print('k and v endpoints',k[[0,1,-2,-1]],v[[0,-1]])
