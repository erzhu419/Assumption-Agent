from laws12 import *
from scipy.optimize import minimize_scalar
u,v=s['panel_a']['u'],s['panel_a']['v']; _,_,keep=calfit('aff'); u,v=u[keep],v[keep]; A=aff(u)
print('affine',np.round(ls(A,v),4));print('cubic',list(combinations_with_replacement(range(4),3)),np.round(ls(aff(u,mon(u,2),mon(u,3)),v)[15:],3).T)
F={'sin':np.sin,'tanh':np.tanh,'sat':lambda q:q/(1+q*q),'soft':lambda q:q/(1+abs(q)),'atan':np.arctan,'exp':lambda q:np.exp(-q*q),'cos':np.cos,'root':lambda q:q/np.sqrt(1+q*q)}
for name,f in F.items():
 for extra in [False,True]:
  def cols(x,k):return aff(x,f(x*k),*( [mon(x,2)] if extra else [] ))
  def res(k):
   C=cols(u,k);return (C@ls(C,v)-v).ravel()
  opt=min([least_squares(res,[k],bounds=(.05,7),max_nfev=90) for k in [.5,1.25,2.15,3.5]],key=lambda o:np.linalg.norm(o.fun)); print(name,extra,opt.x[0],np.sqrt(np.mean(opt.fun**2)))
for j in range(4):
 y=v[:,j]; b=ls(A,y)[1:]; out=[]
 for name,f in F.items():
  for linear in [True,False]:
   def cols(x,p):return aff(*([x] if linear else []),f(x@p[:4]+p[4]))
   def res(p):
    B=cols(u,p);return B@ls(B,y)-y
   opt=min([least_squares(res,np.r_[b*k,off],max_nfev=300,gtol=1e-10) for k in [.5,1.25,2.15] for off in [0,.3]],key=lambda o:np.linalg.norm(o.fun));out.append((np.sqrt(np.mean(opt.fun**2)),name,linear,np.round(opt.x,5)))
 print('single-index',j,sorted(out,key=lambda z:z[0])[:7])
