from probe import *
import sys
sys.path.insert(0,'/holdout_task'); import batch_solver as bs
from scipy.special import erf
u=bs._inverse(x,bs._calibrate(s['panel_a'])); p,v=map(s['panel_b'].get,('u','v'))
g=bs._operator(s['panel_b'],u)
F=[(np.arange(52),np.arange(52,72))]+[(q[:52],q[52:]) for q in (np.random.default_rng(j).permutation(len(x)) for j in range(20))]
for a in [*u.T,*g.T,np.linalg.norm(u,axis=1),np.linalg.norm(g,axis=1)]:
 q=np.argsort(a); F.extend([(q[:52],q[52:]),(q[20:],q[:20])])
funcs={
'rat':(lambda a,k:(a**3+k[1]*a)/(1+k[0]*a*a),[.8,-.08],[0,-.6],[4,.6]),
'sine':(lambda a,k:a+k[1]*np.sin(k[0]*a),[2.9,-.4],[.2,-2],[5,2]),
'erf':(lambda a,k:a+k[1]*erf(k[0]*a),[1.46,-.65],[.2,-3],[5,2]),
'poly':(lambda a,k:a**3+k[0]*a+k[1]*a**5+k[2]*a**7,[-.08,-.45,.13],[-1,-2,-2],[1,2,2]),
'power':(lambda a,k:np.sign(a)*abs(a)**k[0]+k[1]*a,[2.5,-.1],[1.05,-1],[6,1]),
'powrat':(lambda a,k:(np.sign(a)*abs(a)**k[2]+k[1]*a)/(1+k[0]*a*a),[.8,-.08,3],[0,-.6,1.3],[4,.6,6]),
'clip':(lambda a,k:a**3+k[1]*a+k[2]*np.sign(a)*np.maximum(abs(a)-k[0],0)**2,[.6,-.07,-1.2],[.1,-.5,-5],[1.2,.5,1])}

def train(f,spec,i,weight):
 def fit(k,out=False):
  A=aff(f(u[i],k)); B=aff(f(p,k)); W=ls(B,v); H=ls(A,T[i]);
  return (W,H) if out else np.r_[((A@H-T[i])*weight).ravel(),(B@W-v).ravel()]
 opt=least_squares(fit,spec[0],bounds=spec[1:]); return opt.x,*fit(opt.x,True)

def sy(g,y,i,j,mode):
 a,b,c,d=g.T
 A={'quad':poly(g,2),'sparse':np.c_[aff(g),a*c,b*d],'sparse0':np.c_[g,a*c,b*d]}[mode]
 return A[j]@ls(A[i],y[i])

def report(e):
 e=np.array(e); print(np.round(np.r_[e[0],e[:21].max(0),e[21:].max(0)],5))
for name,(f,*spec) in funcs.items():
 for weight in (0,1,3):
  allE=[]; last=[]
  for i,j in F:
   k,W,H=train(f,spec,i,weight); A=aff(f(u,k)); z=A[j]@H; R=np.abs(z-T[j]).mean(0); h=A@W
   allE.append([*[np.mean(abs(sy(h,y,i,j,m)-y[j])) for m in ['quad','sparse','sparse0']],R.mean(),R.max()]);last.append(k)
  print(name,weight,'params',np.round(last[0],4),'panel rms',np.std(aff(f(p,last[0]))@ls(aff(f(p,last[0])),v)-v));report(allE)
print('existing panel operator plus sparse scalar/affine trajectory:')
report([[*[np.mean(abs(sy(g,y,i,j,m)-y[j])) for m in ['quad','sparse','sparse0']],*(lambda R:(R.mean(),R.max()))(abs(aff(g[j])@ls(aff(g[i]),T[i])-T[j]).mean(0))] for i,j in F])
