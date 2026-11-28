exec(open('/tmp/probe.py').read().split('maps={}')[0]); from scipy.optimize import least_squares
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
A=s['panel_a']; f=aff(A['u']); m=ls(f,A['v']);
for _ in range(10):
 i=np.linalg.norm(f@m-A['v'],axis=1).argsort()[:40];m=ls(f[i],A['v'][i]);
u=(x-m[0])/m[1:].diagonal(); print('cal',m); print('svd',np.linalg.svd(z,compute_uv=False));
for rates in ([.22,1.25],[.22,1.25,2.5],[.22,1.25,0.44]):
 F=-np.expm1(-t[:,None]*np.array(rates)); q=ls(F,z.T).T; print('temporal',rates,np.sqrt(np.mean((q@F.T-z)**2)),np.corrcoef(np.c_[q,y].T));
for r in [1,2,3]:
 def resid(p):
  F=-np.expm1(-t[:,None]*np.exp(p)); return (F@ls(F,z[:52].T)-z[:52].T).ravel()
 opt=least_squares(resid,np.log(np.geomspace(.25,1.5,r))); print('learn time',np.exp(opt.x),np.sqrt(np.mean(opt.fun**2)))
F=-np.expm1(-t[:,None]*np.array([.22,1.25])); q=ls(F,z.T).T; b=np.c_[y,q]; folds=[(np.arange(52),np.arange(52,72))]+list(KFold(4,shuffle=True,random_state=27).split(x));
def ev(F,lam=0):
 out=[]
 for i,j in folds:
  pred=F[j]@ls(np.vstack((F[i],np.sqrt(lam)*np.eye(F.shape[1]))),np.vstack((b[i],np.zeros((F.shape[1],3)))))
  out.append(np.r_[abs(pred[:,0]-y[j]).mean(),abs(pred[:,1:]@(-np.expm1(-t[:,None]*np.array([.22,1.25]))).T-z[j]).mean()]);
 return np.array(out)
outs=[]
for kind in ['rat','sin','tanh','atan','root','id']:
 for c in ([0] if kind=='id' else [.06,.11,.17,.22,.28,.36,.45,.60,.8,1.,1.25]):
  w={'rat':lambda:u/(1+c*u*u),'root':lambda:u/np.sqrt(1+c*u*u),'sin':lambda:np.sin(np.sqrt(c)*u)/np.sqrt(c),'tanh':lambda:np.tanh(np.sqrt(c)*u)/np.sqrt(c),'atan':lambda:np.arctan(np.sqrt(c)*u)/np.sqrt(c),'id':lambda:u}[kind]();
  for d in [2,3,4]:
   for lam in [.0001,.003,.025,.12]:
    E=ev(poly(w,d),lam); outs.append((E.max(),kind,c,d,lam,E.round(4).tolist()))
for row in sorted(outs)[:24]:print('cv',row)
w=u/(1+.22*u*u); F=poly(w,3); terms=[()] + [ix for k in range(1,4) for ix in combinations_with_replacement(range(4),k)];
coef=ls(F[:52],b[:52]);print('coefficients');
for j in np.argsort(-np.linalg.norm(coef,axis=1)):print(terms[j],coef[j]);
