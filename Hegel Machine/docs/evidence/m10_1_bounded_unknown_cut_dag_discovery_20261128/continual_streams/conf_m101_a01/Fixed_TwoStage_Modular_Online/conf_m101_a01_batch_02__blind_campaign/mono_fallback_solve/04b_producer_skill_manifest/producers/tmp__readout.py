exec(open('/tmp/probe.py').read().split("print('A',")[0]); from scipy.linalg import solve
from scipy.optimize import minimize_scalar
p=[.59802,.16633,1.1523]; g=np.stack([-np.expm1(-p[0]*t),np.exp(-p[1]*t)*np.sin(p[2]*t)]); h=reg(g.T,tr.T).T
print('amplitude/y correlations',np.corrcoef(np.c_[x,h,y].T).round(3)); print('h ranges',h.min(0),h.max(0))
splits=[(np.arange(52),np.arange(52,72)),(np.arange(16,72),np.arange(16)),(np.r_[0:27,43:72],np.arange(27,43))]
for name,z in [('amp',h),('amp+x',np.c_[h,x]),('x',x)]:
 for d in (1,2,3,4):
  F=poly(z,d); print(name,d,[(lam,[round(np.mean(abs(F[j]@solve(F[i].T@F[i]+lam*np.eye(F.shape[1]),F[i].T@y[i])-y[j])),4) for i,j in splits]) for lam in (1e-5,.01,.1,1)])
 for cap in (.7,1,1.3,1.8,2.5):
  if max(abs(y))>=cap:continue
  F=poly(z,2);print(name,'tanh',cap,[round(np.mean(abs(cap*np.tanh(F[j]@reg(F[i],np.arctanh(y[i]/cap)))-y[j])),4) for i,j in splits])
print('true amplitude readout quadratic',reg(poly(h,2),y));print('y range',y.min(),y.max())
b=s['panel_b']; B=reg(af(b['u']),b['v']); print('B',B)
for name,z in [('x',x),('Ai',(x-w[0])/np.diag(w[1:])),('AiB',af((x-w[0])/np.diag(w[1:]))@B),('AiBi',((x-w[0])/np.diag(w[1:])-B[0])@np.linalg.inv(B[1:]))]:
 F=poly(z,2);print(name,'amplitude polynomial coef',reg(F,h).round(3).T)
