exec(open('/tmp/probe05.py').read().split("for lbl,A in")[0])
from scipy.optimize import minimize_scalar

def basis(z,mode,w,linear):
 f={'tanh':np.tanh,'cos':np.cos,'sin':np.sin}[mode]
 return np.c_[poly(z,1) if linear else np.ones(len(z)),f(z*w)]
def calib(U,V,mode,linear):
 def obj(w):
  A=basis(U,mode,w,linear); R=A@np.linalg.lstsq(A,V,rcond=None)[0]-V;return np.mean(R**2)
 w=minimize_scalar(obj,bounds=(.6,3.3),method='bounded').x
 A=basis(U,mode,w,linear); c=np.linalg.lstsq(A,V,rcond=None)[0]
 return w,lambda Z:basis(Z,mode,w,linear)@c
P=[]
for V,mode in [(v1,'tanh'),(v2,'cos')]:
 for lin in [False,True]:
  w,fn=calib(u,V,mode,lin)
  E=[]
  for i in range(len(u)):
   k=np.arange(len(u))!=i; _,f=calib(u[k],V[k],mode,lin);E.append(np.abs(f(u[~k])-V[~k]).mean())
  print('cal',mode,lin,round(w,5),np.abs(fn(u)-V).mean(),np.mean(E),max(E));P.append(fn(q))
print('q range',q.min(0),q.max(0),'u',u.min(0),u.max(0))
a,b,c,d=P
for tag,z in [('q',q),('a',a),('b',b),('c',c),('d',d),('a,c',np.c_[a,c]),('b,d',np.c_[b,d]),('q,a',np.c_[q,a]),('q,c',np.c_[q,c]),('q,b,d',np.c_[q,b,d]),('tanh_cos',np.c_[np.tanh(2.3*q),np.cos(1.3*q)]),('all',np.c_[q,np.tanh(2.3*q),np.cos(1.3*q)])]:
 for style in ['linear','square','quad','quadblock','limited']:
  if style=='linear': A=poly(z,1)
  if style=='square': A=np.c_[poly(z,1),z*z]
  if style=='quad':
   if z.shape[1]>8:continue
   A=poly(z)
  if style=='quadblock': A=np.column_stack([poly(z[:,i:i+4]) for i in range(0,z.shape[1],4)])
  if style=='limited': A=np.c_[poly(q),z,z*z]
  print('sink',tag,style,report(A,y,True),report(A,T,True))
