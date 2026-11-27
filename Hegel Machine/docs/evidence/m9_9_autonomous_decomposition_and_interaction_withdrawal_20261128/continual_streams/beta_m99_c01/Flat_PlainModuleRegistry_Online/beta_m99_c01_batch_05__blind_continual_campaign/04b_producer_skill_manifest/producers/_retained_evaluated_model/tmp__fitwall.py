import pickle,numpy as np
from scipy.sparse import diags,eye,kron,csr_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
from _verified_upstream import calibrate_upstream
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s=calibrate_upstream(dict(it['study'],reference_tracer_pulses=it['campaign_diagnostics']))
p=s['reference_tracer_pulses']; W=np.stack([v['w_pulse'] for v in p]); Y=np.stack([v['y_pulse_raw'] for v in p]); Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]
nx,ny=s['nx'],s['ny']; ij=np.asarray(s['sensor_ij']); h=[s['L'+d]/(s['n'+d]-1) for d in 'xy']; kap=[s['kappa_'+d] for d in 'xy']; vel=[s['vel_'+d] for d in 'xy']; robin=[s['robin_'+d] for d in 'xy']; scale=np.array([[k[0]/a**2,k[0]/a**2,k[-1]/a**2,k[-1]/a**2] for k,a in zip(kap,h)]); R=np.stack([m.ravel(order='F') for m in s['source_modes']],axis=1); I=eye(nx*ny,format='csc')
def axis(i,t):
 k,v,a=kap[i],vel[i],h[i]; f=(k[:-1]+k[1:])/(2*a*a); t=t*scale[i]; lo=np.r_[-f[:-1]-np.maximum(v[1:-1],0)/a,-t[3]]; up=np.r_[-t[1],-f[1:]-np.maximum(-v[1:-1],0)/a]; return diags([lo,np.r_[t[0],-lo[:-1]-up[1:],t[2]],up],[-1,0,1],format='lil')
def forward(t,wrap=False):
 ax,ay=[axis(i,t.reshape(2,-1)[i,:4]) for i in range(2)]
 if wrap:
  for i,A in enumerate((ax,ay)):
   for j,row,col in [(4,0,-1),(5,-1,0)]: A[row,col]=-t.reshape(2,-1)[i,j]*scale[i,(j-4)*2]
 L=(kron(eye(ny),ax)+kron(ay,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc(); return np.vstack([splu(L+mu*I).solve(R)[ij[:,0]+nx*ij[:,1]] for mu in s['mu_shifts']])
def err(t,ids,wrap=False,project=False):
 e=W@forward(t,wrap).T-Y
 if project:e=e-(e@Q)@Q.T
 return e[ids].ravel()
b=np.ones((2,4)); b[:,::2]+=np.asarray(robin)[:,None]/np.asarray(h)[:,None]/scale[:,::2]; b=b.ravel(); print('robin:',b.reshape(2,4)); print('pulse norms',np.linalg.norm(Y,axis=1));
for name,t in [('open',b),('reflect',np.ones(8))]:
 for pr in [False,True]: print(name,pr,np.sqrt(np.mean(err(t,slice(None),project=pr).reshape(3,-1)**2,axis=1)))
for wrap in [False,True]:
 start=b if not wrap else np.column_stack([b.reshape(2,4),np.zeros((2,2))]).ravel()
 for pr in [False,True]:
  for test in [-1,0,1,2]:
   ids=np.arange(3) if test<0 else np.delete(np.arange(3),test)
   opt=least_squares(err,start,args=(ids,wrap,pr),ftol=1e-11,gtol=1e-12,xtol=1e-11,max_nfev=100)
   print('fit',wrap,pr,test,'rms',np.sqrt(np.mean(err(opt.x,slice(None),wrap,pr).reshape(3,-1)**2,axis=1)),'wall',np.round(opt.x.reshape(2,-1),6),'sv',np.linalg.svd(opt.jac,compute_uv=False)[-1])
