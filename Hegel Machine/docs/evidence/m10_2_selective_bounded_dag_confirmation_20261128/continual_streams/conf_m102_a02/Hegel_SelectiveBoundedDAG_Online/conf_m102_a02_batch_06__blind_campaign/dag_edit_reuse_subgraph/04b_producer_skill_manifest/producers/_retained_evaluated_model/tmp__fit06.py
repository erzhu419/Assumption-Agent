exec(open('/tmp/probe06.py').read().split('for j in range(4):')[0])
import importlib.util
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as comb
def old(j):
 f=importlib.util.spec_from_file_location('old',f'/tmp/hegel_continual_archive/conf_m102_a02_batch_{j:02d}_solver.py'); m=importlib.util.module_from_spec(f);f.loader.exec_module(m);return m
m=old(3); g=old(2)
co=m._fit_calibration(a); r=a['v']-design(a['u'],3)@co
print('calib terms',[(i,co[i].round(4).tolist()) for i in range(len(co)) if np.any(co[i])]); print('calib row residuals',np.sort(np.linalg.norm(r,axis=1)).round(4)); print('outliers',np.where(np.linalg.norm(r,axis=1)>1)[0]);
q=m._invert(x,co); print('q extremes',q.min(0),q.max(0),'reconstruction',np.max(abs(design(q,3)@co-x)))
panelb=lambda u:op._b_features(u,z['p'])@z['w']; B=panelb(q)
for name,h in [('cal',q),('B',B),('B2',panelb(B))]:
 for d in (1,2,3):
  f=design(h,d); print(name,d,*(np.mean(abs(f[52:]@ls(f[:52],v[:52])-v[52:])).round(6) for v in (y,tr)))
for name,h in [('cal',q),('raw',x),('B',B)]:
 yh=g._predict_scalar(h[:52],y[:52],h[52:]); print('gated prior',name,np.mean(abs(yh-y[52:])))
for name,h in [('u',c['u']),('B',panelb(c['u'])),('v1',c['v1'])]:
 for k in ('v1','v2'):
  for d in (1,2):
   f=design(h,d); print('C',name,k,d,np.sqrt(np.mean((f@ls(f,c[k])-c[k])**2)))
