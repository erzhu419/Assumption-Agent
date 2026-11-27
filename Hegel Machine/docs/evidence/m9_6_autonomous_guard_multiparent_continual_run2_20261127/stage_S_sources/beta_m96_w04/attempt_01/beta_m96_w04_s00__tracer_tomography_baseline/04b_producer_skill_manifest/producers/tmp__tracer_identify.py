import pickle,numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
p=pickle.load(open('/holdout_task/public_beta_m96_w04_s00.pkl','rb'))[0]; s,g=p['study'],p['gt']; V=s['V_reaction_2d']; nx,ny=V.shape
Ix,Iy=sp.eye(nx,format='csc'),sp.eye(ny,format='csc'); E=sp.eye(V.size,format='csc'); M=len(s['source_modes']); src=s['source_modes'].reshape(M,-1).T; ij=s['sensor_ij']; obs=ij[:,0]*ny+ij[:,1]; Z=s['Z_nuisance']; P=np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
F0=g['terminal_field_A'].ravel(); S0=g['S_tilde']; fn=np.linalg.norm(F0); sn=np.linalg.norm(S0)
R=s['reaction_x'][:,None]+s['reaction_y']; D=lambda w: sp.diags(np.broadcast_to(w,V.shape).ravel(),format='csc')
def terms(a,delta=0):
 k=s['kappa_'+a]; v=s['vel_'+a]; n=len(k); h=s['L'+a]/(n+delta); r=s['robin_'+a]; b=2*k[1:]*k[:-1]/(k[1:]+k[:-1])/h**2
 diff=sp.diags([-b,np.r_[b,0]+np.r_[0,b],-b],[-1,0,1],format='csc'); bd=np.zeros(n); bd[[0,-1]]=r*k[[0,-1]]/(h*(r*h/2+k[[0,-1]])); adv=sp.diags([-np.maximum(v[1:],0)/h,abs(v)/h,np.minimum(v[:-1],0)/h],[-1,0,1],format='csc')
 inflow=np.zeros(n); inflow[0]=max(0,v[0])/h; inflow[-1]=max(0,-v[-1])/h
 lift=lambda A: sp.kron(A,Iy,format='csc') if a=='x' else sp.kron(Ix,A,format='csc')
 return [lift(A) for A in (diff,sp.diags(bd),adv,sp.diags(inflow))]
T=terms('x')+terms('y')
def residual(A,errs=False):
 U=[splu(A+mu*E).solve(src) for mu in s['mu_shifts']]; F=U[-1]@g['w_opt']; S=np.concatenate([x[obs] for x in U],axis=0); S-=Z@(P@S)
 f=(F-F0)/fn; d=(S-S0)/sn
 return (np.linalg.norm(f),np.linalg.norm(d)) if errs else np.r_[f,d.ravel()]
rows=[]
for dx in (-1,0,1):
 for dy in (-1,0,1):
  X=terms('x',dx); Y=terms('y',dy); base=sum(X[:3]+Y[:3]);
  for b in (0,1):
   for name,Q in [('V',V),('axes',R),('V+axes',V+R),('V-axes',V-R),('2V',2*V)]:
    e=residual(base-b*(X[3]+Y[3])+D(Q),True); rows.append((sum(e),e,dx,dy,b,name))
for r in sorted(rows)[:10]: print('discrete',r)
base=T[:3]+T[4:7]; flex=base+[T[3],T[7],D(V),D(s['reaction_x'][:,None]),D(s['reaction_y']),E]; x0=np.array([1]*6+[0,0,1,0,0,0],float)
def op(x):return sum(t*z for t,z in zip(flex,x))
fit=least_squares(lambda x: residual(op(x)),x0,diff_step=1e-5,max_nfev=45,gtol=1e-11,ftol=1e-11); print('fit [dx,bx,ax,dy,by,ay,inflow_x,inflow_y,V,rx,ry,I]',fit.x, 'errors',residual(op(fit.x),True))
