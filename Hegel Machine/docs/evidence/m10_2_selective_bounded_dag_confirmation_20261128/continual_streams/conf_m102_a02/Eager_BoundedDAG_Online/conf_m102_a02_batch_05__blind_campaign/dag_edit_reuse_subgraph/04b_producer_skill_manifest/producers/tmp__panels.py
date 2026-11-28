exec(open('/tmp/explore.py').read().split('for p in (')[0])
from scipy.optimize import least_squares
for name in ('panel_a','panel_b','panel_c'):
 p=s[name]; u=p['u']; print(name,'u range',u.min(0),u.max(0))
 for k,v in p.items():
  if k=='u': continue
  print(k,'v ranges',v.min(0),v.max(0))
  for w in (u,v):
   dst=v if w is u else u
   for d in (1,2,3):
    F=feat(w,d); B=np.linalg.lstsq(F,dst,rcond=None)[0]
    if d<=2:
     for j in range(4): B[:,j]=least_squares(lambda a:F@a-dst[:,j],B[:,j],loss='cauchy',f_scale=.03).x
    E=np.linalg.norm(F@B-dst,axis=1); print('forward' if w is u else 'reverse',d,np.round(np.quantile(E,[0,.25,.5,.75,.90,1]),4), 'good',sum(E<.08))
    if w is u and d==1: print(B); print('forward residual by row',np.round(E,3))
 if name=='panel_c':
  for expr in (p['v1']+p['v2'],p['v1']-p['v2'],p['v1']*p['v2']):
   for d in (1,2):
    F=feat(u,d); print('combination',d,np.sqrt(np.mean((F@np.linalg.lstsq(F,expr,rcond=None)[0]-expr)**2)))
