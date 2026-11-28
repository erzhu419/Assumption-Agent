exec(open('/tmp/probe.py').read().split('ops={}')[0]); from scipy.optimize import least_squares
from itertools import combinations_with_replacement
p=np.linalg.svd(T,full_matrices=False)[2][0]; p*=np.sign(p[-1]); f=lambda r:-np.expm1(-t*r); R=minimize_scalar(lambda r:np.linalg.norm(p-f(r)/np.linalg.norm(f(r))),bounds=(.15,2),method='bounded').x; print('exp rate',R,'mode discrepancies',p-f(R)/np.linalg.norm(f(R))); P=T@p; print('y, state, corr',np.std(y),np.std(P),np.corrcoef(y,P)[0,1]); B=fit(ext(b['u']),b['v']); C=fit(ext(np.tanh(1.6*c['u'])),c['v1']);
for name,z in [('x',x),('z',(x-cal[1])/cal[0]),('xb',ext(x)@B),('xbi',(x-B[0])@np.linalg.inv(B[1:])),('zb',ext((x-cal[1])/cal[0])@B),('zbi',((x-cal[1])/cal[0]-B[0])@np.linalg.inv(B[1:])),('zc',ext(np.tanh(1.6*(x-cal[1])/cal[0]))@C)]:
 print(name)
 for d in [1,2,3,4]:
  inds=list(combinations_with_replacement(range(4),d)); Q=np.array([np.prod(z[:,i],axis=1) for i in inds]).T
  if d==1: F=ext(Q)
  else: F=np.c_[F,Q]
  if d<=3:
   w=fit(F[:52],np.c_[y,P][:52]); print(d, np.mean(abs(F[:52]@w-np.c_[y,P][:52]),axis=0), np.mean(abs(F[52:]@w-np.c_[y,P][52:]),axis=0),'terms',[(j,[(inds[i],round(w[len(w)-len(inds)+i,j],3)) for i in np.argsort(abs(w[-len(inds):,j]))[-4:]]) for j in [0,1]])
