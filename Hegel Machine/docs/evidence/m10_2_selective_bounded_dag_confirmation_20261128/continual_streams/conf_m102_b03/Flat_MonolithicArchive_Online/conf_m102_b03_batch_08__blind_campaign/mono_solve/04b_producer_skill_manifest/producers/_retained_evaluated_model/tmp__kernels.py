exec(open('/tmp/graph.py').read().split("print('B'")[0])
from scipy.spatial.distance import cdist
from scipy.linalg import solve
Y=np.c_[s['train']['y'],s['train']['traj']]; n=52
print('telemetry',s['telemetry_vec'],'times',s['eval_times']); print('B SV',np.linalg.svd(b[1:],compute_uv=False))
for name,Z in [('z',z),('B',ops['B'](z)),('P',ops['P'](z)),('Q',ops['Q'](z)),('BP',ops['P'](ops['B'](z))),('BQ',ops['Q'](ops['B'](z)))]:
 for other,W in [('none',None),('z',z),('P',ops['P'](z)),('Q',ops['Q'](z))]:
  if W is not None and name==other:continue
  V=Z if W is None else np.c_[Z/Z.std(0),W/W.std(0)]; ds=cdist(V,V,'sqeuclidean'); ds/=np.median(ds)
  best=[]
  for g in [.1,.3,1,3,10]:
   for r in [1e-5,1e-3,.03]:
    K=1+np.exp(-g*ds); pred=K[n:,:n]@solve(K[:n,:n]+r*np.eye(n),Y[:n],assume_a='pos'); e=abs(pred-Y[n:]).mean(0); best.append((e[0]+e[1:].mean(),g,r,e[0],e[1:].mean()))
  print(name,other,np.round(min(best),5))
