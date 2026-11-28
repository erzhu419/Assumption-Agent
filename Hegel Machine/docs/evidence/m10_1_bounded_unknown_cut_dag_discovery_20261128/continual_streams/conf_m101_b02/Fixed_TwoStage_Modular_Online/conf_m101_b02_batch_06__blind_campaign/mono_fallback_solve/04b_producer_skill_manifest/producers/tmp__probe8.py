exec(open('/tmp/scan6.py').read().split('# Screen coordinate')[0])
from scipy.optimize import least_squares
from scipy.special import expit
from scipy.linalg import svd
np.set_printoptions(precision=5,suppress=True)
# The linear and nonlinear coefficient rows in B are proportional: profile one shared transform and one mixing matrix.
funcs={'erf':erf,'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'root':lambda z:z/np.sqrt(1+z*z),'rat':lambda z:z/(1+z*z),'exp2':lambda z:z*np.exp(-z*z),'asinh':np.arcsinh,'oddsat':lambda z:-np.sign(z)*np.expm1(-abs(z)),'xabs':lambda z:z/(1+abs(z))}
found=[]
for name,f in funcs.items():
 for mode in (0,1,2):
  def g(x,p):
   zz=x if mode<2 else x+p[2:6]; k=p[0]; return np.c_[np.ones(len(x)), zz-p[1]*f(k*zz),x*x] if mode==1 else np.c_[np.ones(len(x)),zz-p[1]*f(k*zz)]
  def residual(p):
   z=g(u,p);return (z@ls(z,v)-v).ravel()
  lo=[.03,-8]+([-.5]*4 if mode==2 else []);hi=[7,8]+([.5]*4 if mode==2 else []);
  fitp=min((least_squares(residual,[k,1/k]+([0]*4 if mode==2 else []),bounds=(lo,hi),max_nfev=180) for k in (.25,.8,1.5,2.8,4.2)),key=lambda r:np.linalg.norm(r.fun)); p=fitp.x;z=g(u,p); b=ls(z,v); rr=(z@b-v)/(1-np.sum(z*np.linalg.pinv(z).T,axis=1))[:,None]; ee=np.std(rr); print(name,mode,np.std(fitp.fun),ee,p); found.append((ee,name,mode,p,b))
for d in ['poly','shared']:
 for name,g in [('cub',lambda x,k:x**3),('cubrat',lambda x,k:x**3/(1+k*x*x)),('cubroot',lambda x,k:x**3/np.sqrt(1+k*x*x)),('cubabs',lambda x,k:x**3/(1+k*abs(x))),('quint',lambda x,k:x**3-k*x**5),('expodd',lambda x,k:x*(-np.expm1(-k*x*x)))]:
  def z(x,k):return np.c_[np.ones(len(x)),x,g(x,k)] if d=='poly' else np.c_[np.ones(len(x)),g(x,k)]
  for bounds in [(0.001,1.5),(1.5,8)]:
   k=minimize_scalar(lambda k:np.linalg.norm(z(u,k)@ls(z(u,k),v)-v),bounds=bounds,method='bounded').x; Z=z(u,k); err=Z@ls(Z,v)-v; print(d,name,k,np.std(err),np.std(err/(1-np.sum(Z*np.linalg.pinv(Z).T,axis=1))[:,None]))
# Fit compact readouts of the B-C chain, including cubic skip interactions and low-rank trajectory scores.
j,I0=np.triu_indices(4); pairs=[(0,1),(0,3),(1,2),(2,3)]
cfeat=lambda z:np.c_[np.ones(len(z)),*[z[:,i]*z[:,j] for i,j in pairs]]
C0=lambda z:cfeat(z)@ls(cfeat(uc),pc['v1'])
inner=np.array_split(np.random.default_rng(35).permutation(52),4);splits=[(np.setdiff1d(np.arange(52),te),te) for te in inner]+[(np.arange(52),np.arange(52,72))]; out=[]
prod=lambda z,w:(z[:,:,None]*w[:,None,:]).reshape(len(z),-1)
for ee,n,mode,p,b in sorted(found,key=lambda e:e[0])[:5]+[(0,'orig',0,[],None)]:
 if n=='orig': bb=B(ai)
 else:
  az=ai if mode<2 else ai+p[2:6]; zz=az-p[1]*funcs[n](p[0]*az); bb=(np.c_[np.ones(len(ai)),zz,ai*ai] if mode==1 else np.c_[np.ones(len(ai)),zz])@b
 cc=C0(bb); dd=D(bb); aa=ai
 for zn,zz in [('c',cc),('b+c',np.c_[bb,cc]),('c+d',np.c_[cc,dd])]:
  for sn,sk in [('a',aa),('a3',np.c_[aa,aa**3]),('sat',np.c_[aa,np.tanh(aa)]),('b',bb)]:
   for typ,F in [('base',np.c_[sk,zz,prod(sk,zz)]),('quad',np.c_[sk,zz,prod(sk,zz),q(cc)]),('skip',np.c_[sk,zz,prod(sk,zz),q(aa)]),('both',np.c_[sk,zz,prod(sk,zz),q(cc),q(aa)]),('diag',np.c_[sk,cc,cc*aa,q(cc)])]:
    for l in [.0001,.001,.01,.1]:
     errs=[]
     for tr,te in splits:
      z=(F-F[tr].mean(0))/np.maximum(1e-5,F[tr].std(0)); U,w,V=np.linalg.svd(z[tr],full_matrices=False); pred=Y[tr].mean(0)+(z[te]@V.T*w/(w*w+l))@(U.T@(Y[tr]-Y[tr].mean(0))); E=np.abs(pred-Y[te]); errs.append([E[:,0].mean(),E[:,1:].mean(),E[:,-1].mean()]);
     out.append((n,mode,zn,sn,typ,l,np.mean(errs[:4],axis=0),np.max(errs[:4],axis=0),errs[4]))
for i in (0,1):
 print('inner',i,*sorted(out,key=lambda r:r[-3][i])[:10],sep='\n');print('outer',i,*sorted(out,key=lambda r:r[-1][i])[:6],sep='\n')
