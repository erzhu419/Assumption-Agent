import numpy as np,pickle
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement as cwr, combinations
from scipy.linalg import solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,z=[s['train'][k] for k in ('x','y','traj')]; t=s['eval_times']; n=len(x)
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(f,v):return np.linalg.lstsq(f,v,rcond=1e-9)[0]
a=s['panel_a']; m=np.linalg.norm(aff(a['u'])@ls(aff(a['u']),a['v'])-a['v'],axis=1)<2; da=ls(aff(a['u'][m]),a['v'][m]); q=(x-da[0])/np.diag(da[1:]); b=s['panel_b']; B=ls(aff(b['u']),b['v']); c=s['panel_c']; u=c['u']
gs={'soft':lambda a,k:a/np.sqrt(1+k*a*a),'rat':lambda a,k:a/(1+k*a*a),'tanh':lambda a,k:np.tanh(k*a),'asinh':lambda a,k:np.arcsinh(k*a),'atan':lambda a,k:np.arctan(k*a),'erf':lambda a,k:__import__('scipy').special.erf(k*a),'absrat':lambda a,k:a/(1+k*abs(a)),'rat2':lambda a,k:a*a/(1+k*a*a),'abs':lambda a,k:a*abs(a),'sig':lambda a,k:np.logaddexp(0,k*a),'sinc':lambda a,k:np.sinc(k*a/np.pi),'sine':lambda a,k:np.sin(k*a)}
for k,g in gs.items():
 def F(a,p):return np.c_[aff(a),g(a,p)]
 opts=[]
 for mode in (0,1):
  v=c['v1']; p=least_squares(lambda p:(F(u,p)@ls(F(u,p),v)-v).ravel(),[1.] if mode==0 else [1.]*4,bounds=(.01,10),max_nfev=100).x; w=ls(F(u,p),v); opts.append((p,np.mean(abs(F(u,p)@w-v))))
 print('C1',k,opts)
def G(a):return np.c_[aff(a),a*a/(1+.164*a*a)]
def H(a):return np.c_[aff(a),gs['soft'](a,3.5948)]
p=H(q)@ls(H(u),c['v1']); r=G(q)@ls(G(u),c['v2']); v=aff(q)@B
V=-np.expm1(-t[:,None]*[1.805,.427]); zz=ls(V,z.T).T
print('mode stats',zz.mean(0),zz.std(0));print('mode corr x p r v y',np.round(np.corrcoef(np.c_[zz,q,p,r,v,y].T)[:2,2:],2))
raw=np.c_[q,p,r,v]; print('linear y and state',ls(aff(raw)[:52],np.c_[y,zz][:52]))
fold=[(np.arange(52),np.arange(52,72))]; rng=np.random.default_rng(26)
for _ in range(8):
 ix=rng.permutation(n);fold.append((ix[:52],ix[52:]))
Y=np.c_[y,z]; out=[]
def score(F,name,weights=None):
 if weights is not None: F=F*weights
 for l in (1e-7,1e-5,1e-3,1e-2):
  K=F@F.T; K+=1.; E=[]
  for i,j in fold:
   yc=Y[i].mean(0); yh=K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+l*np.eye(len(i)),Y[i]-yc,assume_a='pos')+yc; e=abs(yh-Y[j]); E.append([e[:,0].mean(),e[:,1:].mean()]);
  E=np.array(E);out.append((E[:,1].max(),E[0,1],E[:,1].mean(),E[:,0].max(),E[0,0],name,l))
for name,W in [('q',q),('p',p),('qr',np.c_[q,r]),('pr',np.c_[p,r]),('pv',np.c_[p,v]),('qpr',np.c_[q,p,r]),('qprv',raw),('v',v),('r',r)]:
 W=(W-W[:52].mean(0))/W[:52].std(0); d=cdist(W,W,'sqeuclidean')/W.shape[1]; P=W@W.T/W.shape[1];
 for deg in (2,3,4):
  K=(1+P)**deg
  for l in (1e-7,1e-4,.01):
   E=[]
   for i,j in fold:
    yh=K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+l*np.eye(len(i)),Y[i],assume_a='pos');e=abs(yh-Y[j]);E.append([e[:,0].mean(),e[:,1:].mean()])
   E=np.array(E);out.append((E[:,1].max(),E[0,1],E[:,1].mean(),E[:,0].max(),E[0,0],name+' poly'+str(deg),l))
 for scale in (.035,.08,.16,.32,.64,1.28):
  for kind,K in [('rbf',np.exp(-d*scale)),('rq',(1+d*scale)**-2)]:
   for l in (1e-7,1e-4,.001,.01):
    E=[]
    for i,j in fold:
     yh=K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+l*np.eye(len(i)),Y[i],assume_a='pos');e=abs(yh-Y[j]);E.append([e[:,0].mean(),e[:,1:].mean()])
    E=np.array(E);out.append((E[:,1].max(),E[0,1],E[:,1].mean(),E[:,0].max(),E[0,0],name+' '+kind+str(scale),l))
for pairs in ((q,p),(q,r),(p,r),(p,v),(q,z[:,[-1]])):
 Q,S=pairs; F=np.column_stack([Q,S]+[Q[:,a,None]*S for a in range(Q.shape[1])]+[np.tanh(Q),np.tanh(S)]);score(F,'interaction '+str([next(j for j,w in enumerate([q,p,r,v,z]) if w is a) if any(w is a for w in [q,p,r,v,z]) else 5 for a in pairs]))
print('state/worst',sorted(out)[:14]);print('state/spec',sorted(out,key=lambda a:a[1])[:12]);print('y/worst',sorted(out,key=lambda a:a[3])[:5]);
