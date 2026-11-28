exec(open('/tmp/stages.py').read().split('train=[',1)[0])
I=[np.arange(52),np.arange(16,72),np.r_[0:27,43:72],np.r_[0:40,56:72]]; J=[np.setdiff1d(np.arange(72),i) for i in I]
print('times/telemetry',t,s['telemetry_vec']); print('trajectory SVD',np.linalg.svd(tr,compute_uv=False)); print('oracle y fit',[(np.std(y-F@reg(F,y)),np.linalg.norm(reg(F,y))) for F in (af(tr),np.c_[poly(x,2),tr],np.c_[poly(x,2),tr,tr**2])]);
Q=poly(x,2); print('cubic coefficient',reg(poly(x,3),y)[15:]);
def ridge(z, y, mode, rng):
 D=af(z); q=poly(z,2); h=reg(q,y)[5:]; H=np.zeros((4,4)); H[np.triu_indices(4)]=h; H=(H+H.T)/2; val,V=np.linalg.eigh(H); P=V[:,[0,-1]]*np.sqrt(abs(val[[0,-1]])); a,b=(P[:,1]+P[:,0]),(P[:,1]-P[:,0]);
 def f(p,z):
  d=af(z); A,B=(d@p[:10].reshape(2,5).T).T
  if 'tanh' in mode: a,b=np.tanh(A),np.tanh(B)
  if 'sin' in mode: a,b=np.sin(A),np.sin(B)
  if 'soft' in mode: a,b=A/np.sqrt(1+A*A),B/np.sqrt(1+B*B)
  if 'atan' in mode: a,b=np.arctan(A),np.arctan(B)
  if 'mixed' in mode: a,b=A,np.tanh(B)
  if 'pol' in mode: a,b=A,B
  if mode.endswith('q'): return np.c_[poly(z,2),a,b,a*b]
  if mode.endswith('p'): return np.c_[d,a,b,a*b,a*a,b*b]
  if mode.endswith('c'): return np.c_[d,a*b,a**3,b**3]
  return np.c_[d,a,b,a*b]
 def residual(p):
  F=f(p,z);return F@reg(F,y)-y
 base=np.r_[0,a,0,b]; starts=[base*k for k in (.5,1,2)]+[base+rng.normal(0,.4,10) for _ in range(3)];
 opt=min((least_squares(residual,p,max_nfev=160,bounds=(-6,6)) for p in starts),key=lambda o:np.linalg.norm(o.fun)); F=f(opt.x,z); w=reg(F,y); return lambda z:f(opt.x,z)@w, np.std(opt.fun), opt.x, w
for name,d in [('x',x),('c1',z['bi1']),('c2',z['bi2'])]:
 for mode in ['tanh','tanhp','tanhq','sin','sinp','soft','softp','atan','mixed','mixedp','polc']:
  fits=[ridge(d[i], y[i],mode,np.random.default_rng(21)) for i in I]; E=[np.mean(abs(p(d[j])-y[j])) for (p,*_),j in zip(fits,J)]; print('FIT',name,mode,np.round(E,5),np.round([g[1] for g in fits],5),flush=True)
  if max(E)<.042: print('BEST',fits[0][2:])
