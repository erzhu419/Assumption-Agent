import importlib.util, itertools, pickle
import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
p='/tmp/hegel_continual_archive/beta_m98_v2_c04_batch_02_solver.py'
spec=importlib.util.spec_from_file_location('archive',p); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; pulses=item['campaign_diagnostics']
Z=s['Z_nuisance']; W=np.column_stack([p['w_pulse'] for p in pulses]); Y=np.column_stack([mod._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in pulses]); Q=np.linalg.qr(Z,mode='reduced')[0]
nx,ny=s['nx'],s['ny']; idx=np.array(s['sensor_ij'])@[1,nx]; rhs=np.einsum('mxy,mp->xyp',s['source_modes'],W).reshape(nx*ny,-1,order='F'); I=eye(nx*ny,format='csc'); V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
def axis(d, kind):
 k,v,L,r=[s[a+'_'+d] for a in ('kappa','vel','L','robin')] if 'L_'+d in s else (s['kappa_'+d],s['vel_'+d],s['L'+d],s['robin_'+d]); n=len(k); h=L/(n-1)
 A=mod._axis(k,v,L,0.).toarray()
 for i,j,t,sgn in ((0,1,n-1,-1),(n-1,n-2,0,1)):
  out=max(sgn*v[i],0)/h; inc=max(-sgn*v[i],0)/h
  if not isinstance(kind,str):
   a,b,c=kind[i!=0]; A[i,j]-=a; A[i,t]-=b; A[i,i]+=a+b+c
   continue
  if kind=='periodic':
   A[i,j]=-(k[i]+k[j])/(2*h*h)-inc; A[i,t]=-(k[i]+k[t])/(2*h*h)-out; A[i,i]=-A[i,j]-A[i,t]; continue
  if kind=='periodic_endpoint':
   t=n-2 if i==0 else 1; A[i,j]=-(k[i]+k[j])/(2*h*h)-inc; A[i,t]=-(k[i]+k[t])/(2*h*h)-out; A[i,i]=-A[i,j]-A[i,t]; continue
  if kind=='robin': A[i,i]+=r/h
  if kind=='face': A[i,j]=-(k[i]+k[j])/(2*h*h); A[i,i]=-A[i,j]
  if kind=='double': A[i,:]*=2
  if kind=='adv_in': A[i,j]-=inc; A[i,i]+=inc
  if kind=='adv_out': A[i,i]+=out
  if kind=='adv_conservative': A[i,i]+=out; A[i,j]-=inc
  if kind=='adv_signed': A[i,i]+=out-inc
  if kind=='adv_abs': A[i,j]-=out+inc; A[i,i]+=out+inc
 return csc_matrix(A)
def predict(ax,ay):
 A=kron(eye(ny),ax,format='csc')+kron(ay,eye(nx),format='csc')+V
 return np.vstack([splu(A+mu*I).solve(rhs)[idx] for mu in s['mu_shifts']])
def profile(R): return R-Q@(Q.T@R)
def scores(R): return np.sqrt(np.mean(R.reshape(3,-1,2)**2,axis=1))
if __name__=='__main__':
 for d in 'xy':
  print(d,'edge kappa',s['kappa_'+d][[0,1,-2,-1]],'edge vel',s['vel_'+d][[0,1,-2,-1]])
 names='reflect robin face double adv_in adv_out adv_conservative adv_signed adv_abs periodic periodic_endpoint'.split(); Ax={k:axis('x',k) for k in names}; Ay={k:axis('y',k) for k in names}; res=[]
 for a,b in itertools.product(names,repeat=2):
  R=predict(Ax[a],Ay[b])-Y; R=profile(R); res.append((np.linalg.norm(R),a,b,scores(R)))
 for val,a,b,e in sorted(res,key=lambda p:p[0])[:12]: print(a,b,val,np.round(e,8).tolist())
 initial=[]
 for dim in 'xy':
  ref=axis(dim,'reflect').toarray(); per=axis(dim,'periodic').toarray(); delta=per-ref
  for i,j,t in ((0,1,-1),(-1,-2,0)):
   initial.extend([-delta[i,j],-delta[i,t],delta[i].sum()])
 for cols in ([0],[1],[0,1]):
  def fun(theta):
   d=theta.reshape(2,2,3); return profile(predict(axis('x',d[0]),axis('y',d[1]))-Y)[:,cols].ravel()
  opt=least_squares(fun,initial,max_nfev=300,gtol=1e-12,ftol=1e-12,xtol=1e-12); d=opt.x.reshape(2,2,3); R=predict(axis('x',d[0]),axis('y',d[1]))-Y
  print('fit cols',cols,'x/y by end: neighbor,wrap,row sum',np.round(d,7).tolist(),'raw',np.round(scores(R),8).tolist(),'profiled',np.round(scores(profile(R)),8).tolist(),'jac sv',np.round(np.linalg.svd(opt.jac,compute_uv=False),8))
