from readout import *
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
bases.pop('cz');bases.pop('dz'); bases['ct']=np.arctanh(np.clip(c/2,-.99,.99));
# Separate linear, polynomial, and saturated single/pair-coordinate effects.
def kernels(bases):
 for nm,a in bases.items():
  for degree in [1,2,3,4,5,6]:
   g=np.c_[np.ones(len(a)),a]/np.sqrt(2); yield nm+'P'+str(degree),(g@g.T)**degree
  dis=cdist(a,a,'sqeuclidean')
  for scale in [.2,.5,1,2,4,8]:
   yield nm+'R'+str(scale),np.exp(-scale*dis)
  for scale in [.3,1,3]:
   g=(a[:,None,:]-a[None,:,:])**2
   for order in [1,2,3]:
    yield nm+'A'+str((scale,order)),np.mean([np.exp(-scale*g[:,:,idx].sum(2)) for idx in combinations(range(4),order)],axis=0)
  for level in [0,1]:
   f,_=feats(a,level); f=(f-f.mean(0))/np.maximum(f.std(0),1e-8); yield nm+'D'+str(level),f@f.T

def norm(K):return K/max(np.mean(K.diagonal())-K.mean(),1e-10)
kd={nm:norm(K) for nm,K in kernels(bases)}
for keys in [('z','tanh'),('z','b','c','d'),('z','b','c','d','tanh','raw')]:
 for typ in ['P','A','R','D','PAR','PARD']:
  kd['+'.join(keys)+typ]=np.mean([K for nm,K in list(kd.items()) if '+' not in nm and nm.split(typ[0])[0] in keys and any(nm.startswith(key+tp) for key in keys for tp in typ)] or [K for nm,K in list(kd.items()) if '+' not in nm and any(nm.startswith(key+tp) for key in keys for tp in typ)],axis=0)
regs=[1e-7,1e-5,.0002,.002,.015,.07,.25]
Yall=np.c_[y,Y]
def ridge(K,T,V,a):
 F=K[np.ix_(T,T)]; P=K[np.ix_(V,T)]; avg=Yall[T].mean(0); F=F-F.mean(0)-F.mean(1)[:,None]+F.mean();P=P-P.mean(1)[:,None]-K[np.ix_(T,T)].mean(0)+K[np.ix_(T,T)].mean();return P@cho_solve(cho_factor(F+a*np.eye(len(T))),Yall[T]-avg)+avg
for j in [0,1]:
 scores=[]
 for nm,K in kd.items():
  for a in regs:
   ee=[]
   for T,V in splits:
    E=abs(ridge(K,T,V,a)-Yall[V]);ee.append(E[:,0].mean() if j==0 else E[:,1:].mean())
   scores.append((np.mean(ee[1:]),ee[0],max(ee[1:]),nm,a))
 print('HEAD',j,'inner',sorted(scores)[:12]); print('HEAD',j,'outer',sorted(scores,key=lambda p:p[1])[:12])
