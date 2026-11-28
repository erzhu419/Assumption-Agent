exec(open('/tmp/recover9.py').read().split('for j in range(4):')[0])
from itertools import combinations
j=np.arange(4); k=(j+2)%4
families={'rat':(lambda x,p: x/(1+p[0]*abs(x)**p[1]),[.7,2.],([.01,.5],[20.,6.])), 'tanh':(lambda x,p:np.tanh(x*p[0]),[1.2],([.01],[8.])), 'sin':(lambda x,p:np.sin(x*p[0]),[2.4],([.01],[8.])), 'oddg':(lambda x,p:x*np.exp(-p[0]*abs(x)**p[1]),[.7,2.],([.01,.5],[10.,6.]))}
fits=[]
for n,(fn,p0,bd) in families.items():
 for m,(gn,q0,be) in families.items():
  def design(w,i,p):
   a,b=w[:,i],w[:,k[i]]
   return np.c_[np.ones(len(w)),a,b,fn(a,p[:len(p0)]),gn(b,p[len(p0):])]
  def obj(p):return np.column_stack([design(u,i,p)@L(design(u,i,p),v[:,i])-v[:,i] for i in j]).ravel()
  fit=least_squares(obj,p0+q0,bounds=(bd[0]+be[0],bd[1]+be[1]));p=fit.x
  fits.append((np.std(fit.fun),n,m,p,np.column_stack([L(design(u,i,p),v[:,i]) for i in j])))
for row in sorted(fits,key=lambda a:a[0])[:8]:print('calfit',*row)
_,n,m,p,c=min(fits,key=lambda a:a[0]);fn,p0,_=families[n];gn,_,_=families[m]
def cal(w):return c[0]+c[1]*w+c[2]*w[:,k]+c[3]*fn(w,p[:len(p0)])+c[4]*gn(w[:,k],p[len(p0):])
def inv(w):
 z=w.copy(); ii=np.arange(4)
 for _ in range(40):
  e=cal(z)-w
  if np.max(abs(e))<1e-11:break
  jac=np.empty((len(z),4,4))
  for i in ii:
   zz=z.copy();zz[:,i]+=1e-6;jac[:,:,i]=(cal(zz)-cal(z))/1e-6
  z-=np.clip(np.linalg.solve(jac,e[...,None])[...,0],-.5,.5)
 print('inverse err',abs(cal(z)-w).max());return z
x=inv(s['train']['x']); cp=s['panel_c']; bp=s['panel_b']; b=A(x)@L(A(bp['u']),bp['v']); h=[]; primitive=[]
for key,fun,p0,bd in [('v1',families['rat'][0],[1.,1.1],([.01,.3],[10.,5.])),('v2',lambda x,p:np.expm1(-p[0]*x*x),[.3],([.01],[4.]))]:
 def des(x,p):return np.c_[A(x),fun(x,p)]
 u=cp['u'];v=cp[key]; opt=least_squares(lambda p:(des(u,p)@L(des(u,p),v)-v).ravel(),p0,bounds=bd);p=opt.x
 print(key,p,np.std(opt.fun)); h.append(des(x,p)@L(des(u,p),v)); primitive.append(fun(x,p))
c,d=h; r,g=primitive
Y=np.c_[s['train']['y'],s['train']['traj']]; A0=np.ones((len(x),1)); tr=np.arange(52);te=np.arange(52,len(x))
def score(f):
 w=L(f[tr],Y[tr]);e=np.abs(f@w-Y);return np.round([e[tr,0].mean(),e[te,0].mean(),e[tr,1:].mean(),e[te,1:].mean()],6)
def quad(x):i,j=np.triu_indices(x.shape[1]);return x[:,i]*x[:,j]
def cross(x,y):return (x[:,:,None]*y[:,None,:]).reshape(len(x),-1)
groups={'x':x,'b':b,'c':c,'d':d,'r':r,'g':g,'xq':quad(x),'bq':quad(b),'cq':quad(c),'dq':quad(d),'xc':cross(x,c),'xd':cross(x,d),'bc':cross(b,c),'bd':cross(b,d),'cd':cross(c,d),'x3':x**3,'c3':c**3,'tanhc':np.tanh(c),'tanhb':np.tanh(b),'tanhd':np.tanh(d)}
for base in [('x',),('c',),('d',),('c','d'),('x','c'),('x','c','d')]:
 f=np.column_stack([A0,*[groups[k] for k in base]]); print('base',base,score(f))
 for key,g0 in groups.items():
  if key not in base: print(key,score(np.c_[f,g0]),end='; ')
 print()
np.savez('/tmp/graph9.npz',Y=Y,**groups)
