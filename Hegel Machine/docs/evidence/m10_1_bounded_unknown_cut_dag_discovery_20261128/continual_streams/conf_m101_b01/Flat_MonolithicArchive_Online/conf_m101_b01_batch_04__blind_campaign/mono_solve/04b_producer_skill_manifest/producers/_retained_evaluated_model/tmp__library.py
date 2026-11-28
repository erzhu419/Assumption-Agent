from shape import *
from scipy.optimize import minimize_scalar
p=s['panel_c']; u=p['u']; H=lambda a,k,p:a/(1+k*abs(a))**p
for extras in (False,True):
 def panel(z,k,p): return np.c_[aff(z),H(z,k,p)] if extras else aff(H(z,k,p))
 def err(par):
  A=panel(u,*np.exp(par)); return (A@ls(A,p['v1'])-p['v1']).ravel()
 opt=least_squares(err,np.log([.47,2]),bounds=(np.log([.01,.25]),np.log([10,10]))); print('v1 shape',extras,np.exp(opt.x),np.std(opt.fun),flush=True)
 if not extras: q=panel(b,*np.exp(opt.x))@ls(panel(u,*np.exp(opt.x)),p['v1']); h=H(b,*np.exp(opt.x));
sp=lambda z:np.logaddexp(0,2.2*z)
r=aff(sp(b))@ls(aff(sp(u)),p['v2']);
from sklearn.linear_model import OrthogonalMatchingPursuit,Lasso
S=[]
def report(n,D):
 for reg in (.00001,.003,.03,.2,1.):
  errors=[]
  for i,j in folds:
   A=D-D[i].mean(0); A/=np.maximum(A[i].std(0),1e-8); yy=y[i]-y[i].mean(); U,sv,V=np.linalg.svd(A[i],full_matrices=False); pred=A[j]@(V.T@((sv/(sv*sv+reg))*(U.T@yy)))+y[i].mean(); errors.append(np.mean(abs(pred-y[j])))
  S.append((max(errors),errors,n,reg))
Z={'b':b,'q':q,'r':r,'h':h,'sp':sp(b)}
for n,z in list(Z.items()):
 report(n+'q',poly(z)); report(n+'c',poly(z,3));
 for m,w in Z.items():
  report(n+m+'q',poly(np.c_[z,w])); report(n+m+'c separate',np.c_[poly(z,3),poly(w,3)]);
for n,center in [('b',b),('q',q),('r',r),('h',h)]:
 for ext in [False,True]:
  z=center; src=np.c_[z,*(z[:,k:k+1]+sg*z[:,j:j+1] for k in range(4) for j in range(k+1,4) for sg in [-1,1])] if ext else z
  for fun in [np.sin,np.tanh,np.arctan,np.abs,lambda a:a/(1+a*a),lambda a:1/(1+a*a),lambda a:np.exp(-a*a)]:
   for scale in [0.5,1,1.7,2.5,4]:
    f=fun(src*scale); report(str((n,ext,fun.__name__,scale)),np.c_[poly(z),f]);
    if not ext: report(str((n,fun.__name__,scale,'quotient')),np.c_[poly(z),*(poly(z)/((1+(scale*z[:,k])**2)[:,None]) for k in range(4))]);
print('best dense',*sorted(S)[:16],sep='\n',flush=True)
# Sparse interaction dictionary with weakly aliased dictionaries kept as separate candidates.
for kind,z in Z.items():
 for n,w in Z.items():
  D0=poly(z)[:,1:]; D1=poly(np.c_[z,w],3)[:,1:];
  for bankname,D in [('jointc',D1),('separatec',np.c_[poly(z,3)[:,1:],poly(w,3)[:,1:]]),('jointq',poly(np.c_[z,w])[:,1:])]:
   for lam in [.001,.003,.01,.025]:
    sc=[]
    for i,j in folds:
     A=D-D[i].mean(0); A/=np.maximum(A[i].std(0),1e-8); f=Lasso(alpha=lam,max_iter=2500,tol=.0002).fit(A[i],y[i]); sc.append(np.mean(abs(f.predict(A[j])-y[j])))
    S.append((max(sc),sc,(kind,n,bankname),lam))
print('best sparse',*sorted([x for x in S if isinstance(x[2],tuple)])[:12],sep='\n')
