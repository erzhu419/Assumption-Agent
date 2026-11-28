exec(open('/tmp/chamber08.py').read().split('ramps=')[0])
print('times', t)
A=np.c_[H,F]; cf=[]
for j in range(Y.shape[1]):
 y=Y[:,j]; ini=np.linalg.lstsq(A,y,rcond=None)[0]; cf.append(least_squares(lambda v:A@v-y,ini,loss='cauchy',f_scale=.009).x)
cf=np.array(cf); print('per-time hold and drift coefficients\n',np.c_[np.r_[t,0],cf,cf[:,1:]@cf[-1,1:]/np.sum(cf[-1,1:]**2)].round(5)); print('held load coefficient versus intrinsic',np.c_[cf[:,0],R@th[:2]].round(5));
# Test coupled reset with a separate time kernel, allowing all drift effects but no extra hold terms unless supported.
ramps={'sqrt':lambda p:2*t/(1+np.sqrt(1+t/np.exp(p[0]))),'asinh':lambda p:np.exp(p[0])*np.arcsinh(t/np.exp(p[0])),'atan':lambda p:np.exp(p[0])*np.arctan(t/np.exp(p[0])),'log':lambda p:np.exp(p[0])*np.log1p(t/np.exp(p[0])),'power':lambda p:np.exp(p[0])*np.expm1(p[1]*np.log1p(t/np.exp(p[0])))/p[1], 'sqrt_quad':lambda p:t/np.sqrt(1+(t/np.exp(p[0]))**2),'sqrt_delay':lambda p:2*t/(1+np.sqrt(1+t/np.exp(p[0])))+p[1]*t*t}
for nm,fun in ramps.items():
 for hold in [0,2]:
  def pred(v):
   return base+np.outer(F@v[:5],np.r_[fun(v[5:]),1.])+(np.outer(H,R@v[-2:]) if hold else 0.)
  start=np.r_[th[2:],np.log(1.),[.5] if nm=='power' else [0.] if nm=='sqrt_delay' else [],np.zeros(hold)]
  def fit(mask): return least_squares(lambda v:((pred(v)-Y)[mask]*W).ravel(),start,loss='soft_l1',f_scale=.012,max_nfev=200).x
  e=np.empty_like(Y)
  for l in np.unique(labs):
   s=labs==l; e[s]=(pred(fit(~s))-Y)[s]
  print(nm,hold,fit(allrows).round(5),'pooled',np.round([np.median(abs(e[:,:-1])),np.median(abs(e[:,-1]))],5),'worst',np.round([max(np.median(abs(e[labs==l,:-1])) for l in np.unique(labs)),max(np.median(abs(e[labs==l,-1])) for l in np.unique(labs))],5),'time',np.median(e,axis=0).round(4))
