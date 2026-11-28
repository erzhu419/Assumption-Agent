exec(open('/tmp/probe.py').read().split('frames=')[0]); from itertools import combinations
for pn in ['panel_a','panel_c']:
 pp=s[pn]; u=pp['u']; print(pn)
 for k,v in pp.items():
  if k=='u':continue
  k0=keep if pn=='panel_a' else np.ones(len(u),bool); U=u[k0]; V=v[k0];
  for d in [1,2,3]:
   pf=PolynomialFeatures(d); H=pf.fit_transform(U); B=np.linalg.lstsq(H,V,rcond=None)[0]; Hinv=np.linalg.pinv(H); R=V-H@B; print(k,d,'res RMS',np.sqrt(np.mean(R**2,0)), 'loo RMS', np.sqrt(np.mean((R/np.maximum(1-np.sum(H*Hinv.T,axis=1),1e-10)[:,None])**2,0))); names=pf.get_feature_names_out();
   if d<=2 or pn=='panel_a':
    for j in range(4): print(j,[(names[i],round(B[i,j],4)) for i in np.argsort(-abs(B[:,j]))[:7]])
  if pn=='panel_c':
   f=np.c_[U,U**2,U**3,np.sin(U),np.cos(U),np.tanh(U),*[U[:,i]*U[:,j] for i,j in combinations(range(4),2)]]
   print('outputs/ranges',V.min(0),V.max(0)); print('top correlations',np.sort(abs(np.corrcoef(np.c_[f,V].T)[:-4,-4:]),axis=0)[-6:]); print('v1-v2',np.corrcoef(np.c_[pp['v1'],pp['v2']].T)[:4,4:])
