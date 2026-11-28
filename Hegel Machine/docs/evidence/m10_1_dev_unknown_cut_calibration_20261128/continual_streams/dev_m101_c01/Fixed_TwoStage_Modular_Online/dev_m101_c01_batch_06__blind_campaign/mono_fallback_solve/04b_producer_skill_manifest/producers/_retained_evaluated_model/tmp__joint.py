from challenge import *
from scipy.optimize import minimize_scalar
x,y,z=[S['train'][k] for k in ('x','y','traj')];t=S['eval_times']; sp=spaces(S,x); ss=list(folds(x));

def amplitude(t,Z):
 def fit(r):
  b=-np.expm1(-r*t); c=Z@b/(b@b);return c,b
 def obj(r):
  c,b=fit(r);return np.sum((Z-c[:,None]*b)**2)
 r=minimize_scalar(obj,bounds=(.3,1.5),method='bounded').x
 return r,*fit(r)

def ridge_rat(X,U,Y,V,W,kind,reg=0.015):
 F=design(X,2); G=design(U,2); d=V.shape[1]; cen=V.mean(0); V=V-cen;W=W-cen
 def den(A,p):
  if kind=='rad':return 1+np.exp(p[0])*np.sum(A*A,axis=1)
  if kind=='diag':return 1+(A*A)@np.exp(p)
  return 1+(A@p[1:]+p[0])**2
 def calc(p,ret=False):
  D=F/den(V,p)[:,None];a=np.linalg.lstsq(np.r_[D,np.eye(F.shape[1])*reg],np.r_[Y,np.zeros(F.shape[1])],rcond=None)[0]
  return G@a/den(W,p) if ret else np.r_[D@a-Y,reg*a,.012*p]
 if kind=='rad':starts=[[-1.8],[-.6]]
 elif kind=='diag':starts=[np.full(d,-2),np.full(d,-.5)]
 else:starts=[np.r_[0,p] for p in np.eye(d)*.35]+[np.full(d+1,.15)]
 b=min((least_squares(calc,p,max_nfev=100) for p in starts),key=lambda o:o.cost)
 return calc(b.x,True)

if __name__=='__main__':
 for i,j in ss:
  ix=np.r_[i,j]; n=len(i); r,amp,b=amplitude(t,z[i]); G=group([sp['b'][ix]],'mixed'); H=group([sp['ab'][ix]],'smooth'); pa=.55*predict(x[i],x[j],amp,G)+.45*predict(x[i],x[j],amp,H)
  R=z[i]-amp[:,None]*b; _,_,vt=np.linalg.svd(R,full_matrices=False); v=vt[0]; cam=R@v; pc=predict(x[i],x[j],cam,G); B=np.c_[amp,cam]; Q=np.c_[pa,pc]; eta=np.linalg.lstsq(np.c_[np.ones(n),B],y[i],rcond=None)[0][1:];
  print('time',round(r,4),'state',*[round(np.abs(p-z[j]).mean(),5) for p in [pa[:,None]*b,pa[:,None]*b+pc[:,None]*v]],'eta',np.round(eta,3),flush=True)
  for shrink in (0,.6,1):
   yh=shrink*Q@eta+(.5*predict(x[i],x[j],y[i]-shrink*B@eta,group([sp['b'][ix]],'smooth'))+.5*predict(x[i],x[j],y[i]-shrink*B@eta,group([sp['ba'][ix]],'mixed')));print('joint',shrink,np.abs(yh-y[j]).mean(),end='; ')
  print();
  for k in ['x','a','ai','b','ba','ab']:
   out=[]
   for mode in ['rad','diag','rank1']:
    yh=ridge_rat(x[i],x[j],y[i],sp[k][i],sp[k][j],mode);out.append(np.abs(yh-y[j]).mean())
   print(k,np.round(out,5),end='; ')
  print(flush=True)
