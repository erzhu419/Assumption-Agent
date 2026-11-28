import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations,product
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True)
ls=lambda X,y:np.linalg.lstsq(X,y,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def calibration(p):
 u,v=p['u'],p['v']; q=.18*np.tanh(1.8*u)-.32*np.sin(2.4*np.roll(u,2,axis=1)); Y=v-q; B=np.stack([ls(aff(u[:,j:j+1]),Y[:,j]) for j in range(4)],1)
 for _ in range(5):
  r=Y-B[0]-u*B[1]; m=np.max(abs(r),1)<.5; B=np.stack([ls(aff(u[m,j:j+1]),Y[m,j]) for j in range(4)],1)
 def f(x):return B[0]+B[1]*x+.18*np.tanh(1.8*x)-.32*np.sin(2.4*np.roll(x,2,axis=1))
 def inv(x):
  z=(x-B[0])/B[1]
  for _ in range(180):z=(x-B[0]-.18*np.tanh(1.8*z)+.32*np.sin(2.4*np.roll(z,2,axis=1)))/B[1]
  return z
 return f,inv
f,_=calibration(s['panel_a'])
def inv(x):
 z=x.copy()
 for _ in range(35):
  e=f(z)-x; d=(f(z+1e-5)-f(z-1e-5))/(2e-5) # only use finite differences per coordinate below
  J=np.stack([(f(z+1e-5*q)-f(z-1e-5*q))/(2e-5) for q in np.eye(4)],-1)
  delta=np.linalg.solve(J,e[...,None])[...,0]; z-=np.clip(delta,-.28,.28)
 return z
x=s['train']['x']; V=np.c_[s['train']['y'],s['train']['traj']]; z=inv(x); print('inverse residual',np.max(abs(f(z)-x)), 'range',z.min(),z.max())
from itertools import product,combinations
from scipy.special import expit
fun={'I':lambda x:x,'S':np.sin,'T':np.tanh,'A':np.arctan,'C':np.cos,'R':lambda x:x/(1+x*x),'H':lambda x:x/np.sqrt(1+x*x)}
ops={}
for panel,col,name in [('panel_b','v','b'),('panel_c','v1','c'),('panel_c','v2','d')]:
 u,v=s[panel]['u'],s[panel][col]; res=[]; print(name,'ranges',u.min(),u.max(),v.min(),v.max(),flush=True)
 for (gn,g),(hn,h) in product(fun.items(),repeat=2):
  if gn=='I' and hn=='I':continue
  for mode in [0,1]:
   def F(u,p):return aff(u+p[0]*g(p[1]*u+p[2]))
   def G(v,p):return v+p[3]*h(p[4]*v+p[5])
   def calc(p,ret=False):
    X=F(u,p); Y=G(v,p); B=ls(X,Y); e=(X@B-Y)/np.maximum(abs(1+p[3]*(h(p[4]*(v+1e-5)+p[5])-h(p[4]*(v-1e-5)+p[5]))/2e-5),.4)
    hat=np.sum(X*np.linalg.pinv(X).T,1)
    return (np.mean(abs(e)/(1-hat[:,None])),p,gn,hn,B) if ret else e.ravel()
   for p0 in [[-.35,2.6,0,.1,1.5,0],[.3,2.2,0,-.1,2.8,.7]][mode:mode+1]:
    opt=least_squares(calc,p0,bounds=([-2,.25,-2,-.4,.25,-2],[2,5,2,.4,5,2]),max_nfev=140);res.append(calc(opt.x,True))
 for a in sorted(res,key=lambda a:a[0])[:6]:print(name,a[:4],flush=True)
 p0={'b':[2.8478,1.5688,.39752],'c':[2.43053,.00054,6],'d':[1.2663,.00545,-.66712]}[name]; g=np.arctan if name=='c' else np.cos
 X=lambda q,p=p0,g=g:aff(q+p[2]*g(p[0]*q+p[1])); W=ls(X(u),v); ops[name]=lambda q,X=X,W=W:X(q)@W
np.savez('/tmp/coords.npz',z=z,b=ops['b'](z),c=ops['c'](z),d=ops['d'](z),V=V)
