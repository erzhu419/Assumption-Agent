import sys,pickle,numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csc_matrix,diags,eye,kron
from scipy.sparse.linalg import splu
sys.path.insert(0,'/tmp/hegel_continual_archive')
import beta_m100_c01_batch_05_solver as old
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=old._calibrate(s); z=s['Z_nuisance']; ns=len(s['sensor_ij']); nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; modes=s['source_modes']; rhs=np.array([m.ravel(order='F') for m in modes]).T; slots=ij[:,0]+nx*ij[:,1]; cap,tot=s['u_box'],s['total_budget']
orig=old._axis
def axis(k,v,h,r,p,form):
 a=orig(k,v,h,r,'P').toarray(); n=len(k)
 for i,d in [(0,1),(n-1,-1)]:
  j=(i+d)%n; q=(i-d)%n
  inside=-a[i,j]; wrap=-a[i,q]
  if form=='local': inside=k[i]/h**2
  if form=='diffonly':
   inside=.5*(k[i]+k[j])/h**2; wrap=.5*(k[i]+k[q])/h**2
  a[i,j]=-inside*p[0]; a[i,q]=-wrap*p[1]; a[i,i]=-a[i,j]-a[i,q]+p[2]*r/h
 return csc_matrix(a)
def trans(p,form):
 ax,ay=[axis(s['kappa_'+a],s['vel_'+a],s['L'+a]/(n-1),s['robin_'+a],p[3*i:3*i+3],form) for i,(a,n) in enumerate(zip('xy',(nx,ny)))]; I=eye(nx*ny,format='csc'); L=kron(eye(ny),ax,format='csc')+kron(ay,eye(nx),format='csc')+diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); return np.vstack([splu(L+d*I).solve(rhs)[slots] for d in s['mu_shifts']])
def residual(A,t):
 Z=z[t]; P=np.linalg.pinv(Z); B=A[t]; b=y[t]; D=B-Z@(P@B); r=b-Z@(P@b); w,_=old._box_simplex(D.T@D+1e-10*np.eye(B.shape[1]),-D.T@r,cap,tot); return A@w+z@(P@(b-B@w))-y
p0=np.tile([1.,1.,0.],2); allrows=np.ones(len(y),bool); blocks=np.arange(len(y))//ns
for form in ('standard','local','diffonly'):
 q=lambda p,t:residual(trans(p,form),t); e=q(p0,allrows); print(form,'fixed rms',np.std(e),'by shift',[round(np.std(e[blocks==f]),7) for f in range(3)]); kw=dict(bounds=([.05,0,-.12]*2,[4,4,3]*2),gtol=1e-10,ftol=1e-10,max_nfev=70); fit=least_squares(lambda p:q(p,allrows),p0,**kw); print('tuned',fit.x.round(5),np.std(fit.fun));
 for scheme in (blocks, np.tile(np.repeat(np.arange(ns//4)%4,4),3)):
  o=[]
  for f in np.unique(scheme):
   t=scheme!=f; a=least_squares(lambda p:q(p,t)[t],p0,**kw); o.append((round(np.mean(q(a.x,t)[~t]**2)**.5,6),a.x.round(2).tolist()))
  print('CV',o)
