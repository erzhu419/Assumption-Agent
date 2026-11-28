from features import *
from itertools import product
from scipy.spatial.distance import cdist
p=s['panel_b']; q=s['panel_c']; tel=s['telemetry_vec']; tt=T.copy()
B=lambda x:D(np.sign(x)*np.maximum(abs(x)-.35,0)**1.5)
C=lambda x:D(x*np.roll(x,-1,axis=1))
E=lambda x:D(np.sqrt(1+.924227*x*x)+.831839*x)
wB=fit(B(p['u']),p['v']);wC=fit(C(q['u']),q['v1']);wE=fit(E(q['u']),q['v2']); ops={'b':lambda x:B(x)@wB,'c':lambda x:C(x)@wC,'e':lambda x:E(x)@wE}
base={'z':z,'b':b,'c':c,'e':e}; fs={}
for a,v in base.items():
 fs[a]=v
 for k in [.22,.5,.8,1,1.25,1.5,2]:
  for opname,op in ops.items():fs[f'{opname}({k}{a})']=op(k*v)
for k in [.22,.5,.8,1,1.25,1.5,2]:
 for r in [.5,.8,1,1.25,1.5]:
  for sg in [-1,1]:
   for a,v in [('z',z),('c',c),('e',e)]:
    v2=k*v+sg*r*b
    fs[f'{k}{a}{sg:+d}*{r}b']=v2
    for opname,op in ops.items():fs[f'{opname}({k}{a}{sg:+d}*{r}b)']=op(v2)
def ev(K,l):
 out=[]
 for i,j in splits:
  pp=K[np.ix_(j,i)]@np.linalg.solve(K[np.ix_(i,i)]+l*np.eye(len(i)),tt[i]);er=abs(pp-tt[j]);out.append([er[:,0].mean(),er[:,1:].mean()])
 return np.asarray(out)
S=[[],[]];
for nm,v in fs.items():
 for mix in [0,1,2]:
  if mix==1: V=np.c_[v,.22*z,b]
  elif mix==2: V=np.c_[v,.22*z]
  else: V=v
  V=V-V[:52].mean(0); V=V/max(.025,np.sqrt(np.mean(V[:52]**2))); L=V@V.T/V.shape[1]; r=np.sqrt(cdist(V,V,'sqeuclidean')/V.shape[1]);
  for kn,K in [('linear',1+L),('quad',(1+L)**2),('cubic',(1+L)**3),('matern',1+L+(1+r+r*r/3)*np.exp(-r)),('rbf',(1+L)**2+np.exp(-.7*r*r))]:
   for reg in ([1e-6,.015] if kn in ['matern','rbf'] else [.0005]):
    A=ev(K,reg)
    for m in range(2):S[m].append((A[:,m].mean(),A[:,m].max(),A[0,m],nm,mix,kn,reg))
for m in range(2):
 print('target',m,'mean/worst/gate');print(*sorted(S[m])[:14],sep='\n');print('gate');print(*sorted(S[m],key=lambda p:p[2])[:8],sep='\n')
print('eval_times',ts)
