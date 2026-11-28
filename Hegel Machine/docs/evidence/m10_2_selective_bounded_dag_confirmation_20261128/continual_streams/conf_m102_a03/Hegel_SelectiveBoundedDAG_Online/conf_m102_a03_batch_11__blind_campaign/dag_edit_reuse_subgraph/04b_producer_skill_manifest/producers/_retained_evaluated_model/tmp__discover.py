exec(open('/tmp/probe.py').read().split("print('traj sv'")[0])
from scipy.optimize import least_squares
P=S['panel_a']; u,v=P['u'],P['v']; good=np.linalg.norm(v,axis=1)<5
print('A calibration',good.sum(),[np.std(v[good,j]-np.c_[af(u[good]),u[good,j]*u[good,(j+1)%4],u[good,j]**3]@fit(np.c_[af(u[good]),u[good,j]*u[good,(j+1)%4],u[good,j]**3],v[good,j])) for j in range(4)])
B=S['panel_b']; cb=fit(af(B['u']),B['v']); print('B coef',cb)
C=S['panel_c']; u=C['u']; c=z@cb[1:]+cb[0]
F={'id':lambda x:x,'atan':lambda x:np.arctan(2.15*x),'atan1':lambda x:np.arctan(1.25*x),'tanh':lambda x:np.tanh(2.15*x),'sin':lambda x:np.sin(2.15*x),'sin1':lambda x:np.sin(1.25*x),'rat':lambda x:x/(1+.22*x*x),'rat1':lambda x:x/(1+1.25*x*x),'cube':lambda x:x**3,'sq':lambda x:x**2}
from itertools import combinations
for key in ('v1','v2'):
 print(key)
 res=[]
 for r in (1,2,3):
  for fs in combinations(F,r):
   a=np.c_[np.ones(len(u)),*[F[f](u) for f in fs]]; co=fit(a,C[key]); res.append((np.linalg.norm(a@co-C[key])/np.sqrt(C[key].size),fs))
 print(sorted(res)[:12]);
 for f in ('atan','sin','rat','rat1'):
  a=af(F[f](u));print(f,fit(a,C[key]).round(3))
print('direct mixed')
for name,a in [('z',z),('b',c)]:
 for f in F:
  score(name+f,af(F[f](a)));score(name+f+'+lin',af(np.c_[a,F[f](a)]))
