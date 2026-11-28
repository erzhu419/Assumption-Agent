exec(open('/tmp/graph_probe.py').read().split("for key in ('v1','v2'):")[0])
from itertools import combinations, combinations_with_replacement
from scipy.optimize import minimize_scalar
Y=s['train']['y']; n=len(Y); rng=np.random.default_rng(18); folds=[np.arange(52,72)]+np.array_split(rng.permutation(n),4)
W={'raw':raw[:n], 'cal':x[:n], 'invB':((x-b[0])@np.linalg.inv(b[1:]))[:n], 'B':(x@b[1:]+b[0])[:n]}
V=s['panel_c']['v1']; U=s['panel_c']['v2']; A=np.c_[np.ones(len(u)),u]
for fn in (lambda z:np.tanh(z),lambda z:np.arctan(z),lambda z:z/np.sqrt(1+z*z),lambda z:z/(1+z*z),lambda z:np.sin(z)):
 for add in (False,True):
  def d(w,c):return np.c_[np.ones(len(w)),w,fn(c*w)] if add else np.c_[np.ones(len(w)),fn(c*w)]
  def e(c):
   D=d(u,c);return np.std(D@np.linalg.lstsq(D,V,rcond=None)[0]-V)
  r=minimize_scalar(e,bounds=(.2,4),method='bounded');print('panel C',fn.__code__.co_names,add,round(r.x,6),round(r.fun,7))
  if r.fun<.009:
   D=d(u,r.x);C=np.linalg.lstsq(D,V,rcond=None)[0]; W['v1'+str(fn.__code__.co_names)+str(add)]=d(W['invB'],r.x)@C

def geometry(w):
 p=list(combinations_with_replacement(range(4),2)); q=np.array([w[:,i]*w[:,j] for i,j in p]).T; B=np.c_[np.ones(n),w,q]; c=np.linalg.lstsq(B,Y,rcond=None)[0]; H=np.zeros((4,4));
 for (i,j),v in zip(p,c[5:]):H[i,j]=H[j,i]=v*(1 if i==j else .5)
 ev,R=np.linalg.eigh(H);return B,R,ev

def library(w,R):
 names=[]; cols=[]
 def add(name,z):names.append(name);cols.append(z)
 phi={'tanh':np.tanh,'sin':np.sin,'cos':np.cos,'atan':np.arctan,'softsign':lambda z:z/np.sqrt(1+z*z),'rat':lambda z:z/(1+z*z),'rat_sq':lambda z:1/(1+z*z)}
 zlist=[(str(i),w[:,i]) for i in range(4)]+[('eig'+str(i),(w@R)[:,i]) for i in range(4)]
 zlist.extend((f'{i}+{c}*{j}',w[:,i]+c*w[:,j]) for i,j in combinations(range(4),2) for c in (-2,-1,-.5,.5,1,2))
 for name,z in zlist:
  for p in (3,4,5,6):add(f'{name}^{p}',z**p)
  for f,fn in phi.items():
   for c in (.6,1,1.3,1.7,2.2,3):add(f'{f}({c}*{name})',fn(c*z))
 for i,j in combinations(range(4),2):
  a,b=w[:,i],w[:,j]
  for c in (.6,1,1.3,1.7,2.2,3):
   for name,fn in phi.items():
    for label,z in [('ab',fn(c*a*b)),('fa*fb',fn(c*a)*fn(c*b)),('a*fb',a*fn(c*b)),('b*fa',b*fn(c*a)),('a/(1+b^2)',a/(1+(c*b)**2)),('b/(1+a^2)',b/(1+(c*a)**2)),('ab/(1+(a+b)^2)',a*b/(1+(c*(a+b))**2)),('ab/(1+(a-b)^2)',a*b/(1+(c*(a-b))**2))]:add(f'{i},{j}:{name}:{c}:{label}',z)
 for ii in combinations_with_replacement(range(4),3):add(str(ii),np.prod(w[:,ii],axis=1))
 return names,np.array(cols).T
for nm,w in W.items():
 B,R,ev=geometry(w); print('\ncoord',nm,'eigen',ev.round(3),'vectors',R.round(2)); names,L=library(w,R); scores=[]
 for te in folds:
  tr=np.setdiff1d(np.arange(n),te); D=B.copy(); hist=[]; m=[]
  for k in range(11):
   c=np.linalg.lstsq(D[tr],Y[tr],rcond=1e-9)[0];m.append(np.mean(abs(D[te]@c-Y[te])));Q,_=np.linalg.qr(D[tr]); r=Y[tr]-D[tr]@c; G=L[tr]-Q@(Q.T@L[tr]); gn=np.linalg.norm(G,axis=0); g=np.divide(abs(r@G),gn,out=np.zeros_like(gn),where=gn>1e-7); j=np.argmax(g);hist.append(names[j]);D=np.c_[D,L[:,j]]
  scores.append(m)
  if len(scores)==1:print('official terms',hist[:7])
 print('official',np.round(scores[0],4),'worst',np.max(scores,axis=0).round(4))
