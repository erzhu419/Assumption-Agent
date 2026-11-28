exec(open('/tmp/spatial_probe.py').read().split('for mode')[0])
from functools import partial
ls=lambda a,b: np.linalg.lstsq(a,b,rcond=1e-10)[0]
z=np.roll(u,-1,axis=1); p=u*z; affine=lambda x:np.c_[np.ones(len(x)),x];
def fit(h, masks=None):
 d=np.c_[np.ones(u.size),u.ravel(),z.ravel()]; w=[]; r=[]
 for j in range(4):
  f=np.c_[d.reshape(-1,4,3)[:,j],h[:,j]]; ix=np.arange(len(u)) if masks is None else masks
  w.append(ls(f[ix],v[ix,j])); r.append(f@w[-1]-v[:,j]);
 return np.array(r).T,np.array(w)
models={'product':lambda k:p,'sin_prod':lambda k:np.sin(k*p),'tanh_prod':lambda k:np.tanh(k*p),'sinh_prod':lambda k:np.sinh(k*p),'rational_prod':lambda k:p/(1+k*p*p),'rational_pair':lambda k:p/(1+k*(u*u+z*z)),'sin_pair':lambda k:np.sin(k*u)*np.sin(k*z),'tanh_pair':lambda k:np.tanh(k*u)*np.tanh(k*z),'sinh_pair':lambda k:np.sinh(k*u)*np.sinh(k*z),'sin_z':lambda k:u*np.sin(k*z),'sin_u':lambda k:z*np.sin(k*u),'tanh_z':lambda k:u*np.tanh(k*z),'tanh_u':lambda k:z*np.tanh(k*u),'cos_sum':lambda k:np.cos(k*(u+z)),'cos_diff':lambda k:np.cos(k*(u-z)),'cos_sum_plus':lambda k:-np.cos(k*(u+z))+np.cos(k*u)+np.cos(k*z),'cos_diff_plus':lambda k:np.cos(k*(u-z))-np.cos(k*u)-np.cos(k*z),'exp_prod':lambda k:np.expm1(k*p)}
for name,g in models.items():
 best=None
 for seed in np.linspace(.15,4,19):
  h=g(seed); ix=np.arange(len(u)); k=seed
  for _ in range(8):
   r,w=fit(g(k),ix); ix=np.argsort(np.linalg.norm(r,axis=1))[:42]
   k=least_squares(lambda kk:fit(g(kk[0]),ix)[0][ix].ravel(),[k],bounds=(.02,6)).x[0]
  r,w=fit(g(k),ix); e=np.linalg.norm(r,axis=1); loss=np.mean(np.sort(e)[:42]**2)**.5
  if best is None or loss<best[0]:best=loss,k,w,e
 loss,k,w,e=best
 print(name,round(loss,8),round(k,5),np.round(np.sort(e)[[36,40,41,42,43,44]],6),np.round(w,4).tolist())
 if name=='product':
  print('panel residuals',np.round(np.c_[u,z,p,v,e],3).tolist() if False else np.round(np.c_[u[:,0],z[:,0],p[:,0],fit(g(k),np.argsort(e)[:42])[0][:,0],e],3))
for key in ['panel_b','panel_c']:
 ux=s[key]['u']; outs=[s[key][k] for k in s[key] if k!='u'];
 for vy in outs:
  print(key,np.round(ls(affine(ux),vy),3),np.linalg.norm(affine(ux)@ls(affine(ux),vy)-vy), 'op errs',[np.linalg.norm(f(ux)-vy) for f in opts]);
