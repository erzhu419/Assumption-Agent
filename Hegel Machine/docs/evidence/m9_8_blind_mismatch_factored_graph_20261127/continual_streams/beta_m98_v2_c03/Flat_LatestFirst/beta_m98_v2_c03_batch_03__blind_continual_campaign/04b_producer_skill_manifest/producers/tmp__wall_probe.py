import importlib.util, itertools, pickle, numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/beta_m98_v2_c03_batch_02_solver.py'); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=it['study']; pulses=it['campaign_diagnostics']; nx,ny=s['nx'],s['ny']; ids=np.ravel_multi_index(np.asarray(s['sensor_ij']).T,(nx,ny),order='F'); V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); I=eye(nx*ny,format='csc'); Ix,Iy=eye(nx,format='csc'),eye(ny,format='csc'); W=np.column_stack([p['w_pulse'] for p in pulses]); rhs=np.column_stack([f.ravel(order='F') for f in s['source_modes']])@W; obs=np.column_stack([old._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in pulses]); Z=s['Z_nuisance'];
base=[]; props=[]
for a,n in [('x',nx),('y',ny)]:
 k=s['kappa_'+a]; v=s['vel_'+a]; h=s['L'+a]/(n-1); base.append(old._axis(k,v,h,0).toarray()); props.extend([(k[0]/h**2,(k[0]+k[1])/2/h**2,(k[0]+k[-1])/2/h**2,s['robin_'+a]/h,max(v[0],0)/h,max(-v[0],0)/h),(k[-1]/h**2,(k[-1]+k[-2])/2/h**2,(k[0]+k[-1])/2/h**2,s['robin_'+a]/h,max(-v[-1],0)/h,max(v[-1],0)/h)])
# Per-wall factors: leakage, inward diffusion, opposite-wall diffusion.
def predict(p):
 axes=[a.copy() for a in base]
 for j,(leak,inner,wrap) in enumerate(np.reshape(p,(4,3))):
  A=axes[j//2]; i,t,b=(0,1,-1) if j%2==0 else (-1,-2,0); k,face,per,robin,_,_=props[j]; A[i,i]=leak*robin+inner*k+wrap*per; A[i,t]=-inner*k; A[i,b]=-wrap*per
 L=kron(Iy,csc_matrix(axes[0]),format='csc')+kron(csc_matrix(axes[1]),Ix,format='csc')+V
 return np.concatenate([splu(L+mu*I).solve(rhs)[ids] for mu in s['mu_shifts']],axis=0)
def report(r): return np.sqrt(np.mean(r**2,axis=0))
if __name__=='__main__':
 print('wall scales',np.round(props,5))
 options=[]
 for j,(k,face,per,rob,a,b) in enumerate(props):
  options.append({'R':(1,1,0),'N':(0,1,0),'P':(0,face/k+b/k,1+a/per),'D':(0,2*face/k,0),'F':(0,face/k,0),'Q':(1,2*face/k,0)})
 scores=[]
 for keys in itertools.product(*options):
  r=predict([d[k] for d,k in zip(options,keys)])-obs; scores.append((max(report(r)),''.join(keys),report(r)))
 print('best discrete',sorted(scores,key=lambda q:q[0])[:12]);
 for label,inds in [('pulse0',[0]),('pulse1',[1]),('both',[0,1])]:
  for leak_only in (True,False):
   def pars(x): return np.c_[x,np.ones(4),np.zeros(4)] if leak_only else x.reshape(4,3)
   x=np.zeros(4) if leak_only else np.tile([0.,1.,0.],4)
   f=least_squares(lambda p:(predict(pars(p))[:,inds]-obs[:,inds]).ravel(),x,ftol=1e-11,xtol=1e-11,gtol=1e-11,max_nfev=120)
   r=predict(pars(f.x))-obs; print(label,'leak-only' if leak_only else 'full-rows',np.round(pars(f.x),5).tolist(),'rmse',report(r),'mean-by-shift',r.reshape(3,-1,2).mean(axis=1).round(5).tolist())
