exec(open('/tmp/readout.py').read().split('for nam,a0')[0])
u=s['panel_a']['u']; v=s['panel_a']['v'];
from itertools import product
for d in range(1,5):
 q=af(polys(u,d)); good=np.ones(len(u),bool)
 for _ in range(9):
  co=fit(q[good],v[good]); r=np.linalg.norm(q@co-v,axis=1); good=r<max(3.5*np.median(r),.0005)
 print('cal polynomial',d,'good',good.sum(),'median',np.median(r),'sorted',np.sort(r)[::4]);
 if d<3: print(co)
qa=lambda u:af(polys(u,2)); u0=u; q=qa(u); co=fit(q,v)
for i in range(20):
 r=np.linalg.norm(q@co-v,axis=1); good=np.argsort(r)[:40]; co=fit(q[good],v[good]);
print('u v first',np.c_[u,v][:10]); print('r current',np.sort(np.linalg.norm(a._forward(u,cal)-v,axis=1)) if hasattr(a,'_forward') else 'n/a')
