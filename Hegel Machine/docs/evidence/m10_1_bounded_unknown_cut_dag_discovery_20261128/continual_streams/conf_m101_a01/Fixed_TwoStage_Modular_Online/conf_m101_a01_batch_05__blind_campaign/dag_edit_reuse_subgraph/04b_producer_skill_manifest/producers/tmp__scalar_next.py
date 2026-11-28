from scan import *
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
ids=np.arange(n); fold=[(ids[:52],ids[52:]),*list(splits(n))[1:]]
X=z['x']; V=(X-X.mean(0))/X.std(0)
A=np.c_[np.ones(len(X)),blocks(X)[1]]
fit=np.linalg.lstsq(A[:n],y,rcond=None)[0]; R=y-A[:n]@fit
print('quadratic coefficients',fit.round(3),'residual std',R.std(),flush=True)

def gp(K,i,j):
 C=cho_factor(K[np.ix_(i,i)],check_finite=False); return y[i].mean()+K[np.ix_(j,i)]@cho_solve(C,y[i]-y[i].mean(),check_finite=False)
def score(K):
 e=np.array([np.mean(abs(gp(K,i,j)-y[j])) for i,j in fold]);return e[0],e[1:].mean(),e.max()
G=1+V@V.T/4; G2=G**2; I=np.eye(len(X)); allr=[]
for k in z:
 W=z[k]; W=(W-W.mean(0))/W.std(0); ds=cdist(W,W,'sqeuclidean'); d1=(W[:,None,:]-W[None,:,:])**2
 for l in (.6,1,1.5,2.2,3.2,5):
  r=ds/l**2; re=np.exp(-r/2); h=np.sqrt(5*r); mat=(1+h+h*h/3)*np.exp(-h)
  add=np.exp(-d1/(2*l*l)).mean(2); prod=add**2
  for typ,C in enumerate([re,mat,prod,.5*prod+.5*re]):
   for a in (.03,.15,.6,2):
    for e in (.00015,.0015,.008,.03):
     m=score(G2+a*C+e*I); allr.append((*m,k,l,typ,a,e))
for m in (0,1,2): print('GP',m,sorted(allr,key=lambda u:u[m])[:5],flush=True)
# Optimize smooth residual kernel hyperparameters within each training fold.
for k in ('x','i','ib','ij','b','bi','x1','ib1','x2','ib2'):
 W=z[k]; W=(W-W.mean(0))/W.std(0); d=(W[:,None,:]-W[None,:,:])**2; err=[]; pars=[]
 for i,j in fold:
  t=y[i]-y[i].mean(); ix=np.ix_(i,i); di=d[ix]; g=G2[ix]; eye=np.eye(len(i));
  def components(p,di,g,eye):
   w=np.exp(p[:4]); scale,amp,noise=np.exp(p[4:]); h=np.exp(-.5*(di@w)); return scale*g+amp*h+noise*eye,h,scale,amp,noise,w
  def obj(p):
   C,h,s,a,e,w=components(p,di,g,eye); L=cho_factor(C,check_finite=False); b=cho_solve(L,t,check_finite=False); z=cho_solve(L,eye,check_finite=False)-np.outer(b,b); grad=np.r_[-.25*a*np.einsum('ij,ij,ijk->k',z,h,di)*w,.5*s*np.sum(z*g),.5*a*np.sum(z*h),.5*e*np.trace(z)]; return .5*t@b+np.log(np.diag(L[0])).sum(),grad
  best=min([minimize(obj,np.r_[[q]*4,-3,-4,-7],jac=True,method='L-BFGS-B',bounds=[(-8,4)]*4+[(-12,2),(-12,2),(-11,-2)]) for q in (-2,0,2)],key=lambda r:r.fun); p=best.x; K=components(p,d,G2,I)[0]; err.append(np.mean(abs(gp(K,i,j)-y[j]))); pars.append(p)
 print('ML',k,np.round([err[0],np.mean(err[1:]),max(err)],5),np.round(pars[0],2),flush=True)
