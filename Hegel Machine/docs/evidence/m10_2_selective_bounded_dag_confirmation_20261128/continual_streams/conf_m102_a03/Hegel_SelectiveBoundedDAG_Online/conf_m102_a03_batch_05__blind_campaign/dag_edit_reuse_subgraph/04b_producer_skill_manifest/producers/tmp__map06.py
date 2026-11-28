exec(open('/tmp/explore06.py').read().split('for nm,F')[0])
from itertools import combinations
u=b['u']; C=s['panel_c']; print('paired inputs',np.max(abs(C['u']-u))); print('C/B first rows',*[C[k][:3] for k in C]); print('B',b['v'][:3]);
i,j=np.triu_indices(4,1); k=np.arange(4)
def pair(z):return z[:,i]*z[:,j]
def cy(z):return z*np.roll(z,-1,axis=1)
def report(u,v):
 for nm,fun in [('lin',a),('cubic',lambda z:np.c_[a(z),z**3]),('cubicpair',lambda z:np.c_[a(z),z**3,pair(z)]),('cubiccy',lambda z:np.c_[a(z),z**3,cy(z)]),('cubicsq',lambda z:np.c_[a(z),z**3,z**2]),('allsq',lambda z:np.c_[a(z),z**3,z**2,pair(z)]),('sin',lambda z:np.c_[a(z),np.sin(z),pair(z)]),('tanh',lambda z:np.c_[a(z),np.tanh(z),pair(z)])]:
  F=fun(u); co=fit(F,v); e=F@co-v; h=(np.linalg.qr(F)[0]**2).sum(1); print(nm,np.sqrt(np.mean(e**2)),np.median(abs(e)/(1-h[:,None])),np.round(co.T,4) if np.sqrt(np.mean(e**2))<.005 else '')
for key,U,V in [('B',u,b['v']),('C1',C['u'],C['v1']),('C2',C['u'],C['v2'])]: print(key);report(U,V)
print('archive metadata files:')
