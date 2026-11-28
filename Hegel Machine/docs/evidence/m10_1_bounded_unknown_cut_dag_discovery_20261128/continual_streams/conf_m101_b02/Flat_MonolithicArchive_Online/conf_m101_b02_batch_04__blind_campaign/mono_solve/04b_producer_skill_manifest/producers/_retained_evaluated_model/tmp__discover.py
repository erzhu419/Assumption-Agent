import pickle
import numpy as np
from scipy.optimize import minimize_scalar,least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=5,suppress=True)
def design(*x): return np.column_stack((np.ones(len(x[0])),*x))
def fit(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
def ref(u,v):
 X=design(u); w=fit(X,v)
 for _ in range(4):
  i=np.argsort(np.linalg.norm(X@w-v,axis=1))[:45]; w=fit(X[i],v[i])
 return w
w=ref(**s['panel_a']);print('reference',w)
funcs={'affine':(lambda x,p:x,0,1),'shrink':(lambda x,p:np.sign(x)*np.maximum(abs(x)-p,0)**1.5,0,.9),'hinge':(lambda x,p:np.sign(x)*np.maximum(abs(x)-p,0),0,.9),'power':(lambda x,p:np.sign(x)*abs(x)**p,1.001,5),'oddexp':(lambda x,p:-np.sign(x)*np.expm1(-p*abs(x)),.005,5),'tanh':(lambda x,p:np.tanh(p*x),.005,5),'sin':(lambda x,p:np.sin(p*x),.005,4),'soft':(lambda x,p:x/np.sqrt(1+p*x*x),.005,5),'rat':(lambda x,p:x/(1+p*x*x),.005,5),'cubic':(lambda x,p:x**3,0,1),'asinh':(lambda x,p:np.arcsinh(p*x),.005,5),'atan':(lambda x,p:np.arctan(p*x),.005,5)}
u=s['panel_b']['u'];v=s['panel_b']['v'];rows=[]
for name,(f,lo,hi) in funcs.items():
 for linear in ([False] if name=='affine' else [False,True]):
  def d(z,p): return design(z,f(z,p)) if linear else design(f(z,p))
  def loss(p):
   g=d(u,p); return (g@fit(g,v)-v).ravel()
  p=minimize_scalar(lambda p:np.linalg.norm(loss(p)),bounds=(lo,hi),method='bounded').x
  g=d(u,p); res=loss(p).reshape(v.shape); h=np.sum(np.linalg.qr(g)[0]**2,axis=1); loo=np.linalg.norm(res/(1-h[:,None])); rows.append((loo,name,linear,p,d,fit(g,v)))
  if np.linalg.norm(res)<.2: print('B',name,linear,p,np.linalg.norm(res),loo)
  if linear and name not in ('cubic',):
   pp=least_squares(loss,np.full(4,p),bounds=(lo,hi)).x
   if np.linalg.norm(loss(pp))<.06: print('B per channel',name,pp,np.linalg.norm(loss(pp)))
# Test rational damping with absolute value and a learned odd mixture; inspect residuals for kinks.
for label,f,initial,bounds in [('blend shrink',lambda x,p:np.sign(x)*np.maximum(abs(x)-p[0],0)**p[1],[.2,1.5],([0,.3],[.85,5])),('damping power',lambda x,p:x/(1+p[0]*abs(x)**p[1]),[.7,2],([.001,.2],[10,6]))]:
 for linear in [False,True]:
  def d(z,p): return design(z,f(z,p)) if linear else design(f(z,p))
  def loss(p):
   g=d(u,p); return (g@fit(g,v)-v).ravel()
  p=least_squares(loss,initial,bounds=bounds).x; g=d(u,p);h=np.sum(np.linalg.qr(g)[0]**2,axis=1);loo=np.linalg.norm(loss(p).reshape(v.shape)/(1-h[:,None])); print('B other',label,linear,p,np.linalg.norm(loss(p)),loo)
print('best B', sorted((r[:4] for r in rows))[:6])
def c1(x): return design(x[:,[0,0,1,2]]*x[:,[1,3,2,3]])
def c2(x,r): return design(x,x*x/np.sqrt(1+r*x*x))
p=s['panel_c']; wc=fit(c1(p['u']),p['v1']); print('C1 reduced residual',np.linalg.norm(c1(p['u'])@wc-p['v1']))
r=minimize_scalar(lambda r: np.linalg.norm(c2(p['u'],r)@fit(c2(p['u'],r),p['v2'])-p['v2']),bounds=(.01,2),method='bounded').x; wd=fit(c2(p['u'],r),p['v2']); print('C2 rate',r)
x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; target=np.column_stack((y,T)); ts=s['eval_times']; split=52
from itertools import combinations_with_replacement
def poly(z,deg): return np.column_stack([np.prod(z[:,i],axis=1) for n in range(1,deg+1) for i in combinations_with_replacement(range(z.shape[1]),n)])
def cross(a,b): return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def ridge(g,alpha):
 A=g[:split]; R=np.eye(A.shape[1])*alpha;R[0,0]=0
 err=np.abs(g[split:]@np.linalg.solve(A.T@A+R,A.T@target[:split])-target[split:]);return np.array([err[:,0].mean(),err[:,1:].mean()])
for aname,a in [('inverse',(x-w[0])@np.linalg.inv(w[1:])),('forward',design(x)@w),('raw',x)]:
 print('coordinates',aname, 'poly baseline',[ridge(design(poly(a,k)),.01) for k in (1,2,3,4)])
 best=[]
 for row in sorted(rows,key=lambda r:r[0])[:6]+[next(r for r in rows if r[1]=='affine')]:
  loo,name,lin,r2,_,wb=row; f=funcs[name][0]; b=(design(a,f(a,r2)) if lin else design(f(a,r2)))@wb;c=c1(b)@wc;e=c2(b,r)@wd
  for i,j,k in [(a,b,c),(a,b,e),(a,c,e),(b,c,e)]:
   for spec,g in [('lin',design(i,j,k)),('dot',design(i,j,k,np.sum(i*j,1),np.sum(i*k,1),np.sum(j*k,1))),('cross12',design(i,j,k,cross(i,j))),('cross13',design(i,j,k,cross(i,k))),('cross23',design(i,j,k,cross(j,k))),('poly2a',design(poly(a,2),j,k,cross(j,k))),('allcross',design(i,j,k,.3*cross(i,j),.3*cross(i,k),.3*cross(j,k)))]:
    for al in [.00001,.001,.02]:
     er=ridge(g,al); best.append((max(er),er.tolist(),name,lin,''.join(('a' if z is a else 'b' if z is b else 'c' if z is c else 'd') for z in (i,j,k)),spec,al))
 print('best:',*sorted(best)[:8],sep='\n')
