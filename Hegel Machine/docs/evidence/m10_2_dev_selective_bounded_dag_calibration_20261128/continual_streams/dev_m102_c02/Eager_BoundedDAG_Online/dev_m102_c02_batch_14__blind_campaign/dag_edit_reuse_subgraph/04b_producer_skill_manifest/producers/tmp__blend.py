exec(open('/tmp/map_cv.py').read().split("cv('raw'")[0])
from scipy.optimize import least_squares
from itertools import combinations
F=lambda zz:n._map_coordinates(s['panel_b'],zz,'sin','v'); G=lambda zz:n._map_coordinates(s['panel_c'],zz,'atan','v1'); H=lambda zz:n._map_coordinates(s['panel_c'],zz,'quad','v2')
B,C,D=F(z),G(z),H(z)
target=np.column_stack([tr,y]); ix=np.arange(52); jx=np.arange(52,72)
features={'z':z,'B':B,'C':C,'D':D,'BC':G(B),'BD':H(B),'CB':F(C),'CD':H(C),'DB':F(D),'DC':G(D),'B*C':B*C,'B*D':B*D,'C*D':C*D,'B+C':B+C,'B+D':B+D,'C+D':C+D,'B+C+D':B+C+D,'C-D':C-D,'B-C':B-C,'B-D':B-D,'B*sigD+C*(1-sigD)':B/(1+np.exp(-D))+C/(1+np.exp(D))}
for src in ['B','C','D']:
 for g in ['z','B','C','D']:
  for dst in ['B','C','D']:
   if src==dst:continue
   l=1/(1+np.exp(-features[g])); features[src+' gate '+g+' '+dst]=features[src]*l+features[dst]*(1-l)
def af(q):return np.c_[np.ones(len(q)),q]
def err(q):
 d=af(q); beta=np.linalg.lstsq(d[ix],target[ix],rcond=None)[0]; r=abs(d[jx]@beta-target[jx]);return np.mean(r[:,:8]),np.mean(r[:,8])
for e,k in sorted((err(v),k) for k,v in features.items())[:20]:print(k,e)
for group in [('B','C','D'),('z','B','C','D'),('B','C','D','B*C','B*D','C*D'),('B','C','D','BC','BD','CB','CD','DB','DC'),('z','B','C','D','BC','BD','CB','CD','DB','DC','B*C','B*D','C*D')]:
 M=np.stack([features[k] for k in group]); print('group',group,'unrestricted',err(np.hstack(M)))
 def residual(p):
  q=af(np.einsum('i,inj->nj',p,M))[ix]; return (q@np.linalg.lstsq(q,tr[ix],rcond=None)[0]-tr[ix]).ravel()
 for init in [np.ones(len(M)),*np.eye(len(M))[:4]]:
  p=least_squares(residual,init,max_nfev=100).x; p=p/np.linalg.norm(p); q=np.einsum('i,inj->nj',p,M); print('fit',p.round(4),np.mean(abs(residual(p))),err(q));
  if err(q)[0]<.05:cv('blend',q)
