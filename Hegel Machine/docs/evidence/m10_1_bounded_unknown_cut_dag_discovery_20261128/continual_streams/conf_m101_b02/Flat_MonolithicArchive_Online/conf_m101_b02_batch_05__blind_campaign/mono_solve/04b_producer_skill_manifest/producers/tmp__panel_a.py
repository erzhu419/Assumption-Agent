from discover import *
u,v=s['panel_a']['u'],s['panel_a']['v']; G=D(u); rng=np.random.default_rng(560)
def robust(g, y, keep=38, starts=150):
 best=None
 for _ in range(starts):
  ix=rng.choice(len(g),max(g.shape[1]+1,8),replace=False); b=fit(g[ix],y[ix]); old=None
  for k in range(10):
   r=(g@b-y)**2; ix=np.sort(np.argpartition(r,keep-1)[:keep]);
   if np.array_equal(old,ix): break
   old=ix; b=fit(g[ix],y[ix]);
  r=(g@b-y)**2; val=np.sort(r)[:keep].mean()
  if best is None or val<best[0]:best=(val,b)
 return best[1]
np.set_printoptions(precision=3,suppress=True,linewidth=150)
for j in range(4):
 y=v[:,j]; b=robust(G,y); r=y-G@b; keep=np.argsort(abs(r))[:39]; print('channel',j,'beta',b,'quantiles',np.sort(abs(r))[[20,30,34,37,40,42,45,47]]); print('rows [u,y,resid]',np.column_stack([u,y,r])[np.argsort(u[:,j])]);
 for name,g in [('quad',D(u,quad(u))),('elementwise',D(u,u*u,u**3)),('cubic',D(u,quad(u),u**3)),('sat',D(u,np.tanh(1.8*u),u*u)),('crossprev',D(u,u*np.roll(u,1,axis=1))),('hinge',D(u,abs(u),shrink(u,.35)))]:
  b=robust(g,y); r=y-g@b; ix=np.argsort(abs(r))[:38]; h=g[ix]@np.linalg.pinv(g[ix]); print(name,'med/38/42',np.sort(abs(r))[[23,37,41]],'loo',mse(r[ix]/np.maximum(1-np.diag(h),.001)))
