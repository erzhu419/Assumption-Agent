from panels import p,np
from scipy.optimize import least_squares
u,v=(p['panel_b'][k] for k in ('u','v'))
def pred(x,y,w,k,q=0):
 h=x+k*np.sin(w*x)+q*x*x
 D=np.c_[np.ones(len(x)),h]; return D
for model in ('shared','pre-square','post-square','post-cross','post-full'):
 def solve(idx,xx):
  x,y=u[idx],v[idx]
  def calc(x,w,k,q,M,b):
   z=pred(x,y,w,k,q if model=='pre-square' else 0)[:,1:]@M
   if model=='post-square':z=z+q*z*z
   if model=='post-cross':z=z+q*z*np.roll(z,-1,axis=1)
   if model=='post-full':z=z+q[0]*z*z+q[1]*z*np.roll(z,-1,axis=1)
   return z+b
  D=pred(x,y,2.8,-.4); C=np.linalg.lstsq(D,y,rcond=None)[0]
  n={'shared':0,'post-full':2}.get(model,1)
  def evaluate(x,t):return calc(x,*t[:2], t[22:] if n==2 else (t[22] if n else 0), t[2:18].reshape(4,4),t[18:22])
  t=least_squares(lambda t:(evaluate(x,t)-y).ravel(),np.r_[2.8,-.4,C[1:].ravel(),C[0],np.zeros(n)],max_nfev=300).x
  return evaluate(xx,t),t
 rows=np.arange(len(u)); yh,t=solve(rows,u); loo=np.array([np.linalg.norm(solve(rows[rows!=j],u[j:j+1])[0][0]-v[j]) for j in rows]); print(model,'w/k/q',t[:2],t[22:],'fit',np.linalg.norm(yh-v)/np.sqrt(len(u)),'LOO',np.sqrt(np.mean(loo**2)),loo.max())
 if model=='shared': print('mixing',t[2:18].reshape(4,4)); print('residual singular values',np.linalg.svd(yh-v,compute_uv=False))
