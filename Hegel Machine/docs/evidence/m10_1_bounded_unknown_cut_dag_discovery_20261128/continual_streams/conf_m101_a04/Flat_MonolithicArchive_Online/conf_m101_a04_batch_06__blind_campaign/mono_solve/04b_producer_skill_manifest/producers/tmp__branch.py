from compose_probe import *
from scipy.linalg import svd
from scipy.spatial.distance import cdist
c,d=C(x),D(x); H=bias(np.c_[c,d]); T=np.c_[y,z]; R=T-H@fit(H[:52],T[:52]); _,S,V=svd(R[:52],full_matrices=False); G=R@V[:3].T
Dct={}
def add(name,f):
 if f.ndim==1:f=f[:,None]
 Dct[name]=f
for k in ['x','r','c','d','b','j','k','l','m','p','q']:add(k,A[k]);
for k,a in list(Dct.items()):
 for g,fn in [('cos',np.cos),('sin',np.sin),('tanh',np.tanh),('abs',np.abs),('soft',lambda a:np.logaddexp(0,a)),('rat',lambda a:a/(1+a*a)),('sqrat',lambda a:1/(1+a*a))]:
  for w in [.5,1.,1.6,2.4,3.2]:add(f'{g}{w}{k}',fn(w*a))
for w in [-2,-1,-.5,.5,1,2]:
 for tag,a,b in [('cd',c,d),('xc',x,c),('xd',x,d),('cl',c,A['l']),('dm',d,A['m'])]:
  u=a+w*b
  for g,fn in [('sq',lambda u:u*u),('cb',lambda u:u**3),('co',lambda u:np.cos(2.4*u)),('si',lambda u:np.sin(2.4*u)),('ta',np.tanh),('ra',lambda u:1/(1+u*u))]:add(f'{g}{tag}{w}',fn(u))
for name,a,b in [('cd',c,d),('xc',x,c),('xd',x,d),('cc',c,c),('dd',d,d),('xx',x,x)]:
 for j in range(4):
  f=np.roll(b,j,axis=1);add(f'{name}mul{j}',a*f);add(f'{name}dot{j}',(a*f).sum(1));add(f'{name}sin{j}',np.sin(2.4*a*f));add(f'{name}cos{j}',np.cos(2.4*a*f))
  for w in [.2,1.,3.]:add(f'{name}rat{j}{w}',a/(1+w*f*f));add(f'{name}inv{j}{w}',f/(1+w*a*a))
for name,a in [('x',x),('r',raw),('cd',np.c_[c,d]),('c',c),('d',d)]:
 for deg in [2,3,4]: add(f'poly{name}{deg}',poly(a,deg)[:,poly(a,deg-1).shape[1]:]);
F=np.column_stack(list(Dct.values())); names=[f'{k}:{j}' for k,a in Dct.items() for j in range(a.shape[1])]; F-=H@fit(H[:52],F[:52]);scale=F[:52].std(0);F/=np.where(scale>1e-5,scale,1)
for i,g in enumerate(G.T):
 ix=np.argsort(abs(F[:52].T@g[:52]))[-14:][::-1];print('mode',i,[(names[j],round(np.corrcoef(F[:52,j],g[:52])[0,1],3)) for j in ix]);
score=[];start=0
for k,a in Dct.items():
 P=F[:,start:start+a.shape[1]];start+=a.shape[1];
 if P.shape[1]>25:continue
 W=fit(P[:52],G[:52]);err=P@W-G;score.append((np.std(err[:52,0]),np.mean(abs(err[52:,0])),k))
print('group',sorted(score)[:25]);print('by gate',sorted(score,key=lambda k:k[1])[:15]);
for shift in [-1,0,1]:
 K0=np.zeros((len(x),len(x))); kern={};
 for key,a in [('x',x),('r',raw),('c',c),('d',d),('cd',np.c_[c,d]),('xcd',np.c_[x,c,d])]:
  f=a/a[:52].std(0); q=cdist(f,f,'sqeuclidean')/f.shape[1];
  for w in [.15,.5,1.5]:kern[key+str(w)]=np.exp(-q*w)
 for i,j in combinations(kern,2):
  if i[0]!=j[0]:kern[i+'+'+j]=(kern[i]+kern[j])*.5
 scores=[]
 for k,K in kern.items():
  for l in [.0001,.003,.04]:
   P=H@fit(H[:52],T[:52]);E=K[:, :52]@np.linalg.solve(K[:52,:52]+l*np.eye(52),T[:52]-P[:52]); M=np.mean(abs((T-P-E)[52:]),0);scores.append((M[0],M[1:].mean(),k,l))
 print('kernel',sorted(scores)[:8], sorted(scores,key=lambda q:q[1])[:5]);break
