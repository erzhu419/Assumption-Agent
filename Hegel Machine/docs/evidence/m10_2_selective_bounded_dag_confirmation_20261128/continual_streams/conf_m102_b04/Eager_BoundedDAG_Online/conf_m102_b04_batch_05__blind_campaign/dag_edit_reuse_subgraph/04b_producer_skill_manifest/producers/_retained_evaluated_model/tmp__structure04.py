exec(open('/tmp/probe04.py').read().split('for p in')[0])
def fit(f,v):
 r=v-f@np.linalg.lstsq(f,v,rcond=None)[0]; return np.sqrt(np.mean(r*r))
for p,k in [('panel_a','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[p]['u'],s[p][k]; g=poly(u,1)
 if p=='panel_a':
  ix=np.arange(len(u));
  for _ in range(12):
   coef=np.linalg.lstsq(g[ix],v[ix],rcond=None)[0]; e=np.linalg.norm(v-g@coef,axis=1); ix=np.argsort(e)[:39]
  print('robust A',coef,np.sort(e)); continue
 for f in [g, np.c_[g,u*u],poly(u,2), np.c_[g,u*u,u**3],np.c_[g,np.sin(u),np.cos(u),np.sin(2*u)]]:
  print(p,k,f.shape,fit(f,v),np.mean([fit(f[np.arange(len(u))!=i],v[np.arange(len(u))!=i]) for i in range(len(u))]));
 print('quadratic coefficients',np.linalg.lstsq(poly(u,2),v,rcond=None)[0]);
print('frozen versus raw corr', np.corrcoef(x.T,q.T)[:4,4:]);print('trajectory modal y correlation',np.corrcoef(np.c_[y,z@np.linalg.svd(z,full_matrices=False)[2][:2].T].T))
