import pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
c=pickle.load(open('/holdout_task/public_beta_m96_w03_s00.pkl','rb'))[0]
s,g=c['study'],c['gt']; nx,ny=s['nx'],s['ny']; N=nx*ny
Z=s['Z_nuisance']; W=np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
i,j=s['sensor_ij'].T; ix=i*ny+j
src=s['source_modes'].reshape(-1,N).T
I=eye(N,format='csc'); V=s['V_reaction_2d'].ravel()

def build(a):
 n,L,k,v,r=[s[key] for key in ('n'+a,'L'+a,'kappa_'+a,'vel_'+a,'robin_'+a)]; h=L/n
 t=2*k[:-1]*k[1:]/(k[:-1]+k[1:])/h**2
 rob=np.zeros(n); rob[[0,-1]]=r*k[[0,-1]]/(h*(r*h/2+k[[0,-1]]))
 D=diags([-t,np.r_[0,t]+np.r_[t,0],-t],[-1,0,1]); A=np.zeros((n,n)); E=[np.zeros(n),np.zeros(n)]
 for q,z in enumerate(v/h):
  p=q-(1 if z>0 else -1)
  if 0<=p<n: A[q,q]+=abs(z); A[q,p]-=abs(z)
  else: E[0][q]=abs(z)
  if q in (0,n-1) and 0<=p<n: E[1][q]=abs(z)
 print(a,'velocity range',v.min(),v.max(),'inflow terms',E[0][[0,-1]],'outflow terms',E[1][[0,-1]])
 lift=(lambda B:kron(B,eye(ny),format='csc')) if a=='x' else (lambda B:kron(eye(nx),B,format='csc'))
 return [lift(B) for B in (D,diags(rob),diags(A.diagonal())+diags([A.diagonal(-1),A.diagonal(1)],[-1,1]),diags(E[0]),diags(E[1]))]
B=build('x')+build('y')+[diags(V,format='csc')]
gt_f=g['terminal_field_A'].ravel(); gt_s=g['S_tilde']; w=g['w_opt']

def calc(p):
 A=sum(t*b for t,b in zip(p,B)); out=[]
 for mu in s['mu_shifts']:
  U=splu(A+mu*I).solve(src); out.append(U[ix])
 samp=np.vstack(out); return samp-Z@(W@samp),U@w

def report(p):
 S,F=calc(p); return np.max(np.abs(S-gt_s)),np.linalg.norm(F-gt_f)/np.linalg.norm(gt_f)
base=np.ones(11); base[[4,9]]=0
for inflow in (0,1,2):
 for outflow in (0,1,2):
  p=base.copy();p[[3,8]]=inflow;p[[4,9]]=outflow
  print('inflow/outflow',inflow,outflow,'errors',report(p))

def fit(ids):
 def unpack(x):
  p=base.copy();p[ids]=x;return p
 def fun(x):
  S,F=calc(unpack(x));return np.r_[((S-gt_s)/np.linalg.norm(gt_s)).ravel(),(F-gt_f)/np.linalg.norm(gt_f)]
 opt=least_squares(fun,base[ids],max_nfev=40,gtol=1e-13,ftol=1e-13,xtol=1e-13)
 p=unpack(opt.x); print('fitted [Dx Rx Ax I O Dy Ry Ay I O V]',p,'errors',report(p))
fit([3,4,8,9]);fit(list(range(11)))
