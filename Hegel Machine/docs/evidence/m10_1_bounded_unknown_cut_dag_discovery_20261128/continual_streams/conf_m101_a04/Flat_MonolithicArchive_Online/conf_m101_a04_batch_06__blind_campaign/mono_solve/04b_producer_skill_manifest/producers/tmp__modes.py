from compose_probe import *
from scipy.linalg import svd
from scipy.optimize import least_squares
c,d=C(x),D(x); H=bias(np.c_[c,d]); W=fit(H[:52],np.c_[y,z][:52]); R=np.c_[y,z]-H@W; U,S,V=svd(R[:52],full_matrices=False); print('linear readout coefficients',np.round(W,4)); print('residual spectrum',S,'directions',np.round(V[:4],5));print('orig spectrum',svd(z[:52],compute_uv=False));
for i in range(4):
 g=R@V[i]; print('mode',i,'most extreme',[(k,np.round(x[k],3),round(g[k],4)) for k in np.argsort(abs(g[:52]))[-5:]])
for l,G in [('c',bias(c)),('d',bias(d)),('cd',H),('x',bias(x))]:
 W=fit(G[:52],z[:52]); T=z-G@W;print(l,'sv',svd(T[:52],compute_uv=False),'traj linear coefficients',np.round(W,3))
fam={'e':lambda k:-np.expm1(-k*t),'r':lambda k:t/(1+k*t),'d':lambda k:t*np.exp(-k*t),'s':lambda k:np.sin(k*t),'q':lambda k:t/(1+k*t*t),'h':lambda k:np.tanh(k*t),'p':lambda k:t**k,'l':lambda k:np.log1p(k*t),'g':lambda k:-np.expm1(-k*t*t),'u':lambda k:t*t/(1+k*t*t)}
for label,T in [('orig',z[:52]),('cdres',R[:52,1:])]:
 for L in [1,2,3]:
  ls=[]
  for keys in combinations_with_replacement(fam,L):
   def Q(p):return np.array([fam[k](v) for k,v in zip(keys,p)]).T
   def res(p):
    a=Q(p);return (a@fit(a,T.T)-T.T).ravel()
   p=least_squares(res,np.linspace(.4,1.7,L),bounds=(.001,8),max_nfev=70).x;ls.append((round(np.std(res(p)),7),''.join(keys),p))
  print(label,L,sorted(ls,key=lambda a:a[0])[:7])
