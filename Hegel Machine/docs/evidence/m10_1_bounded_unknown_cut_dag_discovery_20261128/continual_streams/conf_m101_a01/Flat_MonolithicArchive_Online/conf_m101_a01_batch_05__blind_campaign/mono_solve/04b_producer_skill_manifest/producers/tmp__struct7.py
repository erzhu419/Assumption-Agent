from explore import *
from itertools import product
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares
np.set_printoptions(precision=4,suppress=True,linewidth=160)
folds=[np.arange(52,72),np.arange(56,72)]+[np.random.default_rng(i).permutation(72)[:20] for i in range(8)]
splits=[(np.setdiff1d(np.arange(72),v),v) for v in folds]
F={}
for key,a in [('x',x),('invA',d),('fwdA',x*gain+bias)]:
 for key2,c in [('',a),('B',aug(a)@B),('iB',(a-B[0])@Bi)]:
  F[key+key2]=c
  if key2=='iB':continue
  for name,fn in [('v1',z),('v2',w)]:
   q=fn(c); J= np.vstack([(fn(np.eye(4)[j:j+1]*1e-4)-fn(-np.eye(4)[j:j+1]*1e-4))[0]/2e-4 for j in range(4)])
   for nm,back in [('id',np.eye(4)),('B',B[1:]),('iB',Bi),('jac',np.linalg.inv(J))]:
    for m in [-1.,-.5,-.25,.25,.5,1.,1.5,2.]:F[(key+key2,name,nm,m)]=a+m*(q-fn(np.zeros((1,4))))@back

def run(target):
 res=[]
 for nm,h in F.items():
  h=h/np.mean(h.std(0)); G=h@h.T/4; D=cdist(h,h,'sqeuclidean')/4
  for label,K in [('cubic005',(1+G)**2+.05*G**3),('cubic05',(1+G)**2+.5*G**3),('cubic',(1+G)**3),('quartic',(1+.3*G)**4),('rbf2',np.exp(-D/8)),('rbf1',np.exp(-D/2)),('resid',(1+G)**2+np.exp(-D/2))]:
   for r in [1e-5,3e-4,.003]:
    es=[]
    for tr,te in splits:
     p=K[np.ix_(te,tr)]@np.linalg.solve(K[np.ix_(tr,tr)]+r*np.eye(len(tr)),target[tr]);es.append(np.mean(abs(p-target[te])))
    res.append((np.mean(es),max(es),*es[:2],nm,label,r))
 print('target',target.shape,'best mean',*sorted(res)[:12],sep='\n');print('best 52',*sorted(res,key=lambda r:r[2])[:8],sep='\n')
if __name__=='__main__':
 run(y);run(T)
 tt=s['eval_times']; U,S,V=np.linalg.svd(T,full_matrices=False); vs=V[:2].T
 def trytime(p):
  decay,omega=p
  f=np.exp(-decay*tt)[:,None]*np.c_[np.sin(omega*tt),np.cos(omega*tt)];return f
 for start in [[.2,1.7],[.2,1.],[.2,2.5]]:
  p=least_squares(lambda p:(trytime(p)@ls(trytime(p),vs)-vs).ravel(),start).x; print('time',p,np.linalg.norm(trytime(p)@ls(trytime(p),vs)-vs))
 print('calibrated poly h eigensystems')
 for nm,h in [('x',x),('d',d),('h',d+w(d)@Bi),('B',b)]:
  f=aug(old._poly(h,[1,2,3])); xx=ls(f,np.c_[y,U[:,:2]*S[:2]]); print(nm,np.sqrt(((f@xx-np.c_[y,U[:,:2]*S[:2]])**2).mean(0)),xx[:15].T)
