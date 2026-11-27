import pickle, numpy as np, importlib.util
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
p='/tmp/hegel_continual_archive/dev_beta_m98_00_s00_solver.py'
spec=importlib.util.spec_from_file_location('old',p); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=it['study']; nx,ny=s['nx'],s['ny']; ii=s['sensor_ij']; idx=ii[:,0]+nx*ii[:,1]; axes=[]
for a,n in [('x',nx),('y',ny)]:
 k=s['kappa_'+a]; h=s['L'+a]/(n-1); rob=s['robin_'+a]; axes.append((old._axis(k,s['vel_'+a],h,rob),k,h,rob))
pulses=it['campaign_diagnostics']; ys=np.column_stack([old._calibrate({**s,'y_obs_raw':p['y_pulse_raw']}) for p in pulses]); rhs=np.column_stack([(p['w_pulse'] @ s['source_modes'].reshape(8,-1)).reshape(nx,ny).ravel(order='F') for p in pulses]); rx=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); eye=sparse.eye(nx*ny,format='csc'); ix=sparse.eye(nx); iy=sparse.eye(ny)
def pred(p,model):
 out=[]
 for dim,(base,k,h,rob) in enumerate(axes):
  a=base.tolil(); n=len(k)
  for w,(i,j,l) in enumerate([(0,1,n-1),(n-1,n-2,0)]):
   q=2*dim+w
   if model=='shared': r,nei,wrap=p[0],1.,0.
   elif model=='axis': r,nei,wrap=p[dim],1.,0.
   elif model=='side': r,nei,wrap=p[q],1.,0.
   elif model=='local': r,nei=p.reshape(4,2)[q]; wrap=0.
   else: r,nei,wrap=p.reshape(4,3)[q]
   a[i,i]=r*rob/h+(nei+wrap)*k[i]/h**2; a[i,j]=-nei*k[i]/h**2; a[i,l]=-wrap*k[i]/h**2
  out.append(a.tocsc())
 op=sparse.kron(iy,out[0],format='csc')+sparse.kron(out[1],ix,format='csc')+rx
 return np.vstack([splu(op+mu*eye).solve(rhs)[idx] for mu in s['mu_shifts']])
print('physical scales',[(np.round(k[[0,1,-2,-1]],5),h,rob) for a,k,h,rob in axes]); print('pulse RMS amplitudes',np.sqrt(np.mean(ys**2,axis=0)))
for model,p0 in [('shared',[1]),('axis',[1,1]),('side',[1]*4),('local',[1,1]*4),('wrap',[1,1,0]*4)]:
 lo=[0]*len(p0); hi=[np.inf]*len(p0)
 for train in ([0],[1],[0,1]):
  fit=least_squares(lambda p:(pred(p,model)-ys)[:,train].ravel(),p0,bounds=(lo,hi),gtol=1e-13,ftol=1e-12,xtol=1e-12,max_nfev=100)
  print(model,train,'par',np.round(fit.x,6),'RMS',np.round(np.sqrt(np.mean((pred(fit.x,model)-ys)**2,axis=0)),8),'rank smin',np.linalg.svd(fit.jac,compute_uv=False)[-1])
