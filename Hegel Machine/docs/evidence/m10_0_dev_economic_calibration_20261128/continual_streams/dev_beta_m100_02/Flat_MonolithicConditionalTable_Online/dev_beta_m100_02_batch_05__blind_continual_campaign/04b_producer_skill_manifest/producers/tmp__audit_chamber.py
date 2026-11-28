import importlib.util, pickle
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
p=Path('/holdout_task'); spec=importlib.util.spec_from_file_location('batch_solver',p/'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load((p/'public_batch.pkl').open('rb'))[0]['study']; nx,ny=s['nx'],s['ny']; Z=s['Z_nuisance']; obs=m._calibrate(s); out=m.solve_batch(s)
def dense_axis(k,v,h,r):
 n=len(k); A=np.zeros((n,n));
 for i in range(n):
  if i in (0,n-1):
   j=1 if i==0 else n-2; A[i,j]=-k[i]/h**2; A[i,i]=k[i]/h**2+r/h
  else:
   A[i,i-1]=-(k[i-1]+k[i])/(2*h**2)-max(v[i],0)/h
   A[i,i+1]=-(k[i]+k[i+1])/(2*h**2)-max(-v[i],0)/h
   A[i,i]=-A[i,i-1]-A[i,i+1]
 return A
Ax,Ay=[dense_axis(s['kappa_'+a],s['vel_'+a],s['L'+a]/(s['n'+a]-1),s['robin_'+a]) for a in 'xy']; L=np.kron(np.eye(ny),Ax)+np.kron(Ay,np.eye(nx))+np.diag(s['V_reaction_2d'].flatten(order='F')); assert np.max(abs(L-m._operator(s).toarray()))<1e-10
rhs=np.stack([u.flatten(order='F') for u in s['source_modes']],axis=1); S=[]; I=np.eye(nx*ny)
for mu in s['mu_shifts']:
 U=np.linalg.solve(L+mu*I,rhs); S.extend(U[s['sensor_ij'][:,0]+nx*s['sensor_ij'][:,1]])
S=np.asarray(S); B=s['total_budget']; cap=s['u_box']; sr=s['source_ridge']; R=s['nuisance_ridge']
def problem(mask):
 z=Z[mask]; d=np.column_stack((S[mask],obs[mask])); P=np.linalg.solve(z.T@z+R*np.eye(z.shape[1]),z.T@d); d=d-z@P; A,y=d[:,:-1],d[:,-1]; return A,y,P
A,y,P=problem(slice(None)); w=out['weights']; c=out['kkt_certificate']; g=A.T@(A@w-y)+sr*w; nu=c['nu_sum']; lo=np.asarray(c['lam_lower']); hi=np.asarray(c['lam_upper']); J=.5*(np.linalg.norm(A@w-y)**2+sr*(w@w)); D=J-g@w+nu*B+cap*np.minimum(0,g-nu).sum()
err=[abs(w.sum()-B),max(0.,-w.min(),w.max()-cap),np.max(abs(g-nu-lo+hi)),np.max(abs(lo*w)),np.max(abs(hi*(cap-w))),abs(J-c['primal_objective']),abs(D-c['dual_lower_bound']),abs(J-D),np.max(abs((U@w).reshape((nx,ny),order='F')-out['terminal_field']))]; assert np.max(err)<1e-10 and np.min(lo)>=0 and np.min(hi)>=0 and np.array_equal(c['weights'],w); print('budget/box/stationarity/complementarity/objective/dual/gap/field:',err)
err=[]
for i in range(0,len(obs),4):
 tr=np.ones(len(obs),bool); tr[i:i+4]=False; A,y,P=problem(tr); w0,_=m._box_simplex(A.T@A+sr*np.eye(len(w)),-A.T@y,cap,B); err.append(np.sqrt(np.mean((S[~tr]@w0+Z[~tr]@(P[:,-1]-P[:,:-1]@w0)-obs[~tr])**2)))
print('Leave-rosette-out RMSE pooled and worst:',np.sqrt(np.mean(np.square(err))),max(err))
rng=np.random.default_rng(120); worst=0.
for total in (0., .42, 1.26, 1.4, 2.94, 3.36):
 for _ in range(20):
  Q=rng.normal(size=(8,8)); H=Q.T@Q+.04*np.eye(8); c=rng.normal(size=8)*3; u,v=m._box_simplex(H,c,.42,total); d=H@u+c-v; violation=max(abs(u.sum()-total),-u.min(),u.max()-.42,np.max(np.maximum(d,0)*u),np.max(np.maximum(-d,0)*(.42-u))); worst=max(worst,violation)
assert worst<1e-9; print('worst stress feasibility/complementarity:',worst)
sw=s['sensor_calibration_sweeps']; f,e,b,coef=m._estimate(sw); x=sw['ref_rosette']; raw=sw['raw_rosette']; re=f(x)-raw; good=np.max(abs(raw),axis=1)<5; size=np.max(abs(re),axis=1); print('shared coefficients:',np.round(coef,5)); print('largest clean residual rows:');
for i in np.flatnonzero(good)[np.argsort(size[good])[-9:]]: print(i, np.round(x[i],3),np.round(re[i],5))
