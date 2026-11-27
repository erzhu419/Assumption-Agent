import sys,pickle,itertools,importlib.util
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares

def load(b):
 p=f'/tmp/hegel_continual_archive/beta_m98_v2_c02_batch_{b:02d}_solver.py'; sp=importlib.util.spec_from_file_location('prior',p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
m=load(6)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; pulses=item['campaign_diagnostics']; nx,ny=s['nx'],s['ny']; N=nx*ny
W=np.stack([p['w_pulse'] for p in pulses],1)
Y=np.stack([m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in pulses],1).reshape(-1,len(s['sensor_ij']),len(pulses))
rows=s['sensor_ij']@[1,nx]; modes=s['source_modes'].reshape(-1,N,order='F').T; rhs=modes@W

def axis(a,p):
 k,v=s['kappa_'+a],s['vel_'+a]; h=s['L'+a]/(len(k)-1); r=s['robin_'+a]; n=len(k)
 M=m._axis(k,v,h).tolil()
 for j,(selfscale,robscale,seamscale) in enumerate(np.asarray(p).reshape(2,3)):
  i,q,t=(0,1,n-1) if j==0 else (n-1,n-2,0)
  inward=k[i]/h**2*selfscale
  seam=(.5*(k[i]+k[t])/h**2+max(v[i] if j==0 else -v[i],0)/h)*seamscale
  M[i,i]=inward+seam+r/h*robscale; M[i,q]=-inward; M[i,t]=-seam
 return M.tocsc()

def pred(p):
 ax,ay=axis('x',p[:6]),axis('y',p[6:]); L=sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); I=sparse.eye(N,format='csc')
 return np.array([splu(L+mu*I).solve(rhs)[rows] for mu in s['mu_shifts']])

def scores(e): return (np.sqrt(np.mean(e**2)), np.sqrt(np.mean(e**2,axis=1)).max())

if __name__=='__main__':
 defs=[]
 for a in 'xy':
  k,v=s['kappa_'+a],s['vel_'+a]; h=s['L'+a]/(len(k)-1)
  for j,(i,q) in enumerate([(0,1),(-1,-2)]):
   per=(.5*(k[i]+k[q])+h*max(-v[i] if j==0 else v[i],0))/k[i]
   defs.append({'N':(1,0,0),'R':(1,1,0),'P':(per,0,1),'D':(2,0,0),'B':(2,2,0)})
 fit=[]
 for labels in itertools.product(defs[0],repeat=4):
  p=np.concatenate([d[l] for d,l in zip(defs,labels)]); e=pred(p)-Y
  fit.append((scores(e),''.join(labels),p))
 for sr,lab,p in sorted(fit)[:12]: print('discrete',lab,sr,p.round(5))
 for init in [sorted(fit)[0][2], np.tile([1.,0.,0.],4)]:
  for train in [np.s_[:,:,:],np.s_[:,:,0],np.s_[:,:,1]]:
   sol=least_squares(lambda p:(pred(p)-Y)[train].ravel(),init,ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=200)
   print('continuous',sol.nfev,scores(pred(sol.x)-Y),sol.x.reshape(-1,3).round(7), 'singular',np.linalg.svd(sol.jac,compute_uv=False).round(6))
