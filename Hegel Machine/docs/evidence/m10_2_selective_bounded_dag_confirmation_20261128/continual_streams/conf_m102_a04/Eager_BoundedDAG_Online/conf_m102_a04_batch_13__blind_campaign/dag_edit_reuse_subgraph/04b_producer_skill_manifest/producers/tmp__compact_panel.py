exec(open('/tmp/stress_node00.py').read().split('for k in families:')[0])
from scipy.special import erf
families.update({'asinh':np.arcsinh,'erf':erf,'sqrt':lambda z:z/np.sqrt(1+z*z)})
def fit_compact(x,y,g,start):
 def f(z,q): return np.c_[np.ones(len(z)),terms(z),g(q[0]*z)+q[1]*z]
 def res(q):
  A=f(x,q); return (A@np.linalg.lstsq(A,y,rcond=1e-10)[0]-y).ravel()
 q=least_squares(res,start,bounds=([.08,-6],[6,6]),gtol=1e-12).x; A=f(x,q); c=np.linalg.lstsq(A,y,rcond=1e-10)[0]
 return q,c,lambda v:f(v,q)@c
for k,g in families.items():
 q,c,pred=fit_compact(u,y,g,[1.1,.0]); e=[]
 for i in range(len(u)):
  m=np.arange(len(u))!=i; e.extend((fit_compact(u[m],y[m],g,[1.1,0])[2](u[~m])-y[~m]).ravel())
 print(k,q,'fit',np.sqrt(np.mean((pred(u)-y)**2)),'loo',np.sqrt(np.mean(np.square(e))),'max',max(abs(np.array(e))))
 if k in ('atan','asinh','sqrt'): print(c.round(5))
