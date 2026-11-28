from probe import *
import sys
sys.path.insert(0,'/holdout_task'); import batch_solver as bs
from scipy.optimize import minimize_scalar
u=bs._inverse(x,bs._calibrate(s['panel_a'])); px,py=map(s['panel_b'].get,('u','v')); g=bs._operator(s['panel_b'],u)
print('study keys',list(s),'train keys',list(tr),'telemetry',s['telemetry_vec'])
folds=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in [np.random.default_rng(j).permutation(len(x)) for j in range(20)]]
for v in [*u.T,*g.T,np.linalg.norm(u,axis=1),np.linalg.norm(g,axis=1)]:
 p=np.argsort(v); folds.extend([(p[:52],p[52:]),(p[20:],p[:20])])
Y=np.c_[y,T]
def cv(A,lam=.03):
 E=[]
 for i,j in folds:
  sc=np.maximum(.01,np.sqrt(np.mean(A[i]**2,0))); a,b=A[i]/sc,A[j]/sc
  R=np.abs(b@np.linalg.solve(a.T@a+lam*np.eye(a.shape[1]),a.T@Y[i])-Y[j]).mean(0)
  E.append([R[0],R[1:].mean(),R[1:].max()])
 E=np.array(E); return np.round(np.r_[E[0],E[:21].max(0),E[21:].max(0)],5)
def panel(A):
 W=ls(A,py); H=(np.linalg.qr(A,mode='reduced')[0]**2).sum(1); return np.sqrt(np.mean(((A@W-py)/np.maximum(.01,1-H[:,None]))**2)),W
tele=float(s['telemetry_vec'][0]); models=[('current',g)]; fs={'cuberat':lambda q,c:q**3/(1+c*q*q),'tanh':lambda q,c:np.tanh(c*q),'erf':lambda q,c:__import__('scipy').special.erf(c*q),'sin':lambda q,c:np.sin(c*q),'cuberat6':lambda q,c:q**3/(1+c*q**6)}
for name,f in fs.items():
 for extra in ('none','linear','interaction','square'):
  def F(z,c):
   a=aff(f(z,c)); return {'none':a,'linear':np.c_[a,z],'interaction':np.c_[a,z,z*np.roll(z,-1,axis=1)],'square':np.c_[a,z,z*z]}[extra]
  cs=np.linspace(.03,3.3,50); ii=np.argmin([panel(F(px,c))[0] for c in cs]); c=minimize_scalar(lambda c:panel(F(px,c))[0],bounds=(cs[max(0,ii-1)],cs[min(ii+1,len(cs)-1)]),method='bounded').x
  for k in ([c,tele] if name=='cuberat' else [c]):
   e,W=panel(F(px,k)); h=F(u,k)@W; print('operator',name,extra,round(k,4),round(e,6),cv(poly(h,2))); models.append((f'{name}/{extra}/{k:.4f}',h))
for name,h in models:
 if name!='current' and not name.startswith(('cuberat/none','cuberat/linear','erf/linear','sin/linear')):continue
 pairs=np.column_stack([h[:,i]+sg*h[:,j] for i in range(4) for j in range(i+1,4) for sg in [-1,1]])
 for label,A in [('affine',aff(h)),('q2',poly(h,2)),('q3',poly(h,3)),('q2 plus cubes',np.c_[poly(h,2),h**3]),('q2 plus pair cubic',np.c_[poly(h,2),pairs**3]),('q2 plus rational',np.c_[poly(h,2),h/(1+tele*h*h)]),('q2 plus latent',np.c_[poly(h,2),u])]:
  print('readout',name,label,cv(A))
