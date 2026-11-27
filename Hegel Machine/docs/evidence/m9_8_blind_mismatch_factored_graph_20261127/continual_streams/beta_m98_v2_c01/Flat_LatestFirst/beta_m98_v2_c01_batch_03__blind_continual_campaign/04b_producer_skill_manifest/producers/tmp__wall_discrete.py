import pickle, importlib.util, numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('base','/tmp/hegel_continual_archive/beta_m98_v2_c01_batch_02_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=p['study']; ds=p['campaign_diagnostics']; Y=np.stack([m._calibrate(dict(s,y_obs_raw=d['y_pulse_raw'])) for d in ds],axis=1); W=np.stack([d['w_pulse'] for d in ds],axis=1)
x,y=s['nx'],s['ny']; I=eye(x*y,format='csc'); ij=s['sensor_ij']; rows=ij[:,0]+x*ij[:,1]; rhs=np.stack([v.ravel(order='F') for v in np.tensordot(W.T,s['source_modes'],axes=1)],axis=1)
ks=[s['kappa_'+a] for a in 'xy']; vs=[s['vel_'+a] for a in 'xy']; hs=[s['Lx']/(x-1),s['Ly']/(y-1)]; rs=[s['robin_'+a] for a in 'xy']; V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
def pred(c,l,v=(0,0,0,0), face=False):
 A=[]
 for j,(k,vv,h,r) in enumerate(zip(ks,vs,hs,rs)):
  a=m._axis(k,vv,h,0).tolil()
  for t,i in enumerate((0,len(k)-1)):
   u=i+(1 if t==0 else -1); kk=.5*(k[i]+k[u]) if face else k[i]; e=c[2*j+t]*kk/h**2+v[2*j+t]*abs(vv[i])/h
   a[i,u]=-e; a[i,i]=e+l[2*j+t]*r/h
  A.append(a.tocsc())
 L=kron(eye(y),A[0],format='csc')+kron(A[1],eye(x),format='csc')+V
 return np.vstack([splu(L+mu*I).solve(rhs)[rows] for mu in s['mu_shifts']])
def err(a): return np.sqrt(np.mean(np.square(a-Y),axis=0))
from itertools import product
q=np.linalg.qr(s['Z_nuisance'])[0]
def axis(j, kind):
 k,v,h,r=ks[j],vs[j],hs[j],rs[j]; n=len(k); A=m._axis(k,v,h,0).tolil()
 for i,u,t in ((0,1,-1),(n-1,n-2,1)):
  if kind in ('R','RL','RR'):
   if kind=='R' or (kind=='RL' and i==0) or (kind=='RR' and i!=0): A[i,i]+=r/h
   continue
  if kind=='N': continue
  if kind=='D':
   A[i,i]=2*k[i]/h**2; A[i,u]=-2*k[i]/h**2; continue
  if kind=='E':
   A[i,i]=k[i]/h**2+r/h; A[i,u]=-k[i]/h**2; continue
  if kind=='I':
   A[i,i]=2*k[i]/h**2; continue
  w=n-1-i
  if kind in ('Pfull','Pdiff','Pseam','Pavg','Pshift','Pdouble'):
   inner=(k[i]+k[u])*.5/h**2 if kind in ('Pfull','Pdiff','Pavg') else k[i]/h**2
   seam=(k[i]+k[w])*.5/h**2 if kind in ('Pfull','Pdiff','Pavg','Pseam','Pdouble') else k[i]/h**2
   if kind=='Pfull': inner+=max(t*v[i],0)/h; seam+=max(-t*v[i],0)/h
   if kind=='Pavg': inner+=max(t*v[i],0)/h; seam+=max(-t*v[i],0)/h; seam*=.5
   if kind=='Pdouble': seam*=2
   if kind=='Pshift': w=n-2 if i==0 else 1
   A[i,u]=-inner; A[i,w]=-seam; A[i,i]=inner+seam
 return A.tocsc()
def eval_pair(ax,ay):
 L=kron(eye(y),ax,format='csc')+kron(ay,eye(x),format='csc')+V
 a=np.vstack([splu(L+mu*I).solve(rhs)[rows] for mu in s['mu_shifts']]); r=a-Y
 return err(a),np.sqrt(np.mean((r-q@(q.T@r))**2,axis=0)),np.sqrt(np.mean(r.reshape(3,32,2)**2,axis=1))
kinds=['N','R','RL','RR','D','I','Pfull','Pdiff','Pseam','Pavg','Pshift','Pdouble']
axes=[{k:axis(j,k) for k in kinds} for j in range(2)]
res=[]
for a,b in product(kinds,repeat=2):
 e,f,sh=eval_pair(axes[0][a],axes[1][b]); res.append((max(e),a,b,e,f,sh))
for _,a,b,e,f,sh in sorted(res,key=lambda row:row[0])[:12]: print(a,b,'direct',e,'profiled',f,'shift',sh.round(7).tolist())
for j in range(2): print('axis',j,'k edges',ks[j][[0,1,-2,-1]],'velocity edges',vs[j][[0,1,-2,-1]])
ref=s['sensor_calibration_sweeps']['ref_rosette']; raw=s['sensor_calibration_sweeps']['raw_rosette']; a=np.c_[np.ones(len(ref)),ref]; test=np.arange(len(ref))%5==0; coef=np.linalg.lstsq(a[~test],raw[~test],rcond=None)[0]; print('held-sweep rmse by channel',np.sqrt(np.mean((a[test]@coef-raw[test])**2,axis=0))); print('pulse weights',W.T)
