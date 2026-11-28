import pickle,numpy as np
from scipy.optimize import minimize_scalar,least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['panel_b']['u']; y=s['panel_b']['v']; n,d=x.shape
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
A=aff(x)
def score(F):
 r=F@ls(F,y)-y; h=(np.linalg.svd(F,full_matrices=False)[0]**2).sum(1); return np.sqrt(np.mean(r*r)), np.sqrt(np.mean((r/(1-h)[:,None])**2))
funcs={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'asinh':np.arcsinh,'sqrt':lambda z:z/np.sqrt(1+z*z),'inv3':lambda z:np.sinh(np.arcsinh(z)/3),'poly3':lambda z:z**3,'poly5':lambda z:z**5}
for name,f in funcs.items():
 for add in (False,True):
  def F(g): return np.c_[A,f(g*x)] if add else aff(f(g*x))
  grid=np.linspace(.15,5,100); vals=[score(F(g))[0] for g in grid]; j=np.argmin(vals); g=minimize_scalar(lambda g:score(F(g))[0],bounds=(grid[max(j-1,0)],grid[min(j+1,99)]),method='bounded').x
  print('pre',name,add, 'g',round(g,4),'fit/loo',score(F(g)),'at 1.25',score(F(1.25)))
print('affine map and nonlinear coefficients'); print(ls(A,y)); print(ls(np.c_[A,x**2,x**3],y)); print('affine residual first 8', (y-A@ls(A,y))[:8]); print('channel extrema',y.min(0),y.max(0))
for name,f in funcs.items():
 rows=[]
 for t in y.T:
  b=ls(A,t); best=None
  for g in [.4,1.25,2.5,4.]:
   for sign in [-1,1]:
    def resid(q): return A@q[:5]+q[5]*f(A@q[6:])-t
    p=least_squares(resid,np.r_[b,sign*.3,g*b/max(np.linalg.norm(b[1:]),.01)],max_nfev=800)
    if best is None or p.cost<best.cost: best=p
  h=(np.linalg.svd(best.jac,full_matrices=False)[0]**2).sum(1); rows.append((np.sqrt(np.mean(best.fun**2)),np.sqrt(np.mean((best.fun/(1-h))**2)),best.x[5],np.linalg.norm(best.x[7:])))
 print('post residual',name,np.round(rows,5))
