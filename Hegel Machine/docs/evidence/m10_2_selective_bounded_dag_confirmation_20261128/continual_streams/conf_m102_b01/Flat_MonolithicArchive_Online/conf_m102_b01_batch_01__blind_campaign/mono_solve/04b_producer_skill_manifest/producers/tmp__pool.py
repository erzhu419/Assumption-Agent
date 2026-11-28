from refine import *
from scipy.optimize import minimize_scalar

def get(x,y,idx):
 def relu(x):return np.maximum(x,0)
 funcs={k:f for k,f in fs.items() if k not in ['id','cubic','exp','cbrt']}
 funcs.update({'logistic':lambda x:expit(x)-.5,'gauss':lambda x:x*np.exp(-x*x),'sat':lambda x:x/(1+abs(x)), 'sinc':lambda x:np.sinc(x/np.pi), 'inv7':lambda x:np.sinh(np.arcsinh(x)/7), 'atanh':lambda x:np.arctanh(np.clip(x,-.9999,.9999)), 'r3':lambda x:x/(1+abs(x)**3), 'pow3':lambda x:x**3,'pow5':lambda x:x**5})
 out=[]
 for k,f in funcs.items():
  for degree in [1,3,5]:
   def D(z,p,k=k,f=f,degree=degree):
    g=p[0]; h=f(z*g); w=h+p[1]*z
    if degree>=3:w=w+p[2]*z**3
    if degree>=5:w=w+p[3]*z**5
    return aff(w)
   def err(p):
    d=D(x,p);return (d@ls(d,y)-y).ravel()
   best=None
   A=aff(x)
   for g in [.35,.65,1.1,1.6,2.5,3.5,5]:
    h=f(x*g); E=np.c_[A,h]; b=ls(E,y); p0=np.r_[g,np.sum(b[1:5]*b[5:])/max(1e-12,np.sum(b[5:]**2)),np.zeros(degree//2)];
    try:o=least_squares(err,p0,bounds=(np.r_[.04,np.full(len(p0)-1,-np.inf)],np.r_[15,np.full(len(p0)-1,np.inf)]),max_nfev=150)
    except ValueError:continue
    if best is None or np.linalg.norm(o.fun)<np.linalg.norm(best.fun):best=o
   if best is None:continue
   o=best; d=D(x,o.x); m=ls(d,y); J=np.c_[np.kron(d,np.eye(4)),o.jac]; lev=np.sum(J*np.linalg.pinv(J).T,axis=1); out.append((np.std(o.fun/(1-lev)),np.std(o.fun),k,degree,o.x,lambda z,d=D,p=o.x,m=m:d(z,p)@m))
 return sorted(out,key=lambda q:q[0])
if __name__=='__main__':
 for idx,(k,j) in enumerate([('panel_b','v'),('panel_c','v1'),('panel_c','v2')]):
  print(k,j,*[v[:-1] for v in get(s[k]['u'],s[k][j],idx)[:10]],sep='\n',flush=True)
