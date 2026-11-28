import pickle,numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from operators import ls,lin
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; n=len(x); states=dict(np.load('/tmp/cvops.npz')); folds=[(np.arange(52),np.arange(52,72))]; rng=np.random.default_rng(123)
for _ in range(5):
 ids=rng.permutation(n);folds.append((ids[:52],ids[52:]))
p=s['panel_c']; u=p['u']; z=u*1.6
ops={'lin':u,'square':u*u,'cube':u**3,'tanh':np.tanh(z),'atan':np.arctan(z),'sin':np.sin(z),'cos':np.cos(z),'root':z/np.sqrt(1+z*z),'rational':z/(1+z*z),'exp':np.exp(-z*z),'cross':np.stack([u[:,i]*u[:,j] for i in range(4) for j in range(i+1,4)],axis=1),'radial':u/(1+.2*(u*u).sum(1,keepdims=True)),'rowgate':np.tanh(z)/(1+.2*(u*u).sum(1,keepdims=True))}
for name in ['v1','v2']:
 base=['tanh'] if name=='v1' else ['lin','square']; v=p[name]; out=[]
 for extra in [[],*[[a] for a in ops if a not in base],*[[a,b] for i,a in enumerate(ops) for b in list(ops)[i+1:] if a not in base and b not in base]]:
  U=np.column_stack([np.ones(len(u)),*[ops[k] for k in base+extra]]); C=ls(U,v); H=(U*np.linalg.pinv(U).T).sum(1); E=v-U@C; out.append((np.sqrt(np.mean((E/np.maximum(1-H[:,None],1e-8))**2)),np.std(E),extra))
 print('panel',name,sorted(out)[:8]);print('panel core coefficient',ls(np.column_stack([np.ones(len(u)),*[ops[k] for k in base]]),v).round(3))

def gp(z,i,j,fam,kind):
 d=(z[:,None]-z[None,:])**2; dt=d[i][:,i]; dv=d[j][:,i]; p=lin(z); P=p[i]; q=p[j]; yy=y[i]; I=np.eye(len(i));
 def kernel(a,par):
  D=a*np.exp(par[:4]); D=D.sum(2)
  if fam=='rbf':return np.exp(-D)
  if fam=='matern':
   R=np.sqrt(5*D);return (1+R+R*R/3)*np.exp(-R)
  return (1+D)**-2
 def m(par,ret=False):
  K=kernel(dt,par)+np.exp(par[4])*I; F=cho_factor(K); kP=cho_solve(F,P); ky=cho_solve(F,yy); c=np.linalg.solve(P.T@kP,P.T@ky); r=yy-P@c; alpha=ky-kP@c
  if ret:return q@c+kernel(dv,par)@alpha,par
  scale=r@alpha; ll=(len(i)-P.shape[1])*np.log(scale)+2*np.log(F[0].diagonal()).sum()+np.linalg.slogdet(P.T@kP)[1]; return ll+kind*np.sum((par[:4]+1.5)**2)
 bounds=[(-5,2)]*4+[(-12,-2)]; opt=minimize(m,[-1.5]*4+[-6],bounds=bounds,method='L-BFGS-B'); return m(opt.x,True)
for key in ['X','A','AB','Ab','BA','Ba','C','AC','BC','CA','bC']:
 for fam in ['rbf','matern','rq']:
  for kind in [.15,1.5]:
   pred=[gp(states[key],i,j,fam,kind) for i,j in folds]; e=[abs(p-y[j]).mean() for (p,_),(i,j) in zip(pred,folds)]; print('gp',key,fam,kind,np.round(e,4),round(np.mean(e),4),pred[0][1].round(2))
