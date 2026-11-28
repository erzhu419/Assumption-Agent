import pickle, runpy, numpy as np
from scipy.optimize import least_squares
root='/holdout_task/'
s=pickle.load(open(root+'public_batch.pkl','rb'))[0]['study']; n=runpy.run_path(root+'node_00.py'); f=runpy.run_path(root+'_frozen_nodes/node_01.py'); fit=n['_fit']; A=n['_affine']; inv=n['_inverse']; b=fit(A(s['panel_b']['u']),s['panel_b']['v']); a=n['_panel_a'](s['panel_a']); x=s['train']['x']; y=s['train']['y']; tr=s['train']['traj']; q=inv(x,a); ib=A(q)@b; p=s['panel_c']; u=p['u']; V=p['v1']
# Compare shared and channel-specific saturating bases, always refitting on the retained rows.
def soft(x): return x/(1+np.abs(x))
for name,op,start in [('softsign',soft,[.83,-7.3]),('asinh',np.arcsinh,[3.5,-2.75])]:
 for separate in (False,True):
  def fit1(U,T):
   def design(W,t): return A(np.c_[W,op(t[0]*W)] if separate else W+t[1]*op(t[0]*W))
   def residual(t):
    d=design(U,t); return (d@fit(d,T)-T).ravel()
   r=least_squares(residual,start[:1] if separate else start,bounds=([.05] if separate else [.05,-30],[15] if separate else [15,30])); c=fit(design(U,r.x),T)
   return r.x,lambda W:design(W,r.x)@c
  pars,_=fit1(u,V); errs=[]
  for j in range(len(u)):
   k=np.arange(len(u))!=j; _,F=fit1(u[k],V[k]); errs.append(F(u[j:j+1])[0]-V[j])
  print('v1',name,separate,pars,'LOO RMSE',np.sqrt(np.mean(np.square(errs))))
warp=lambda z: n['_logcosh'](1.1*z)/1.1
m=fit(A(u+warp(u)),p['v2']); print('v2 noise RMSE',np.sqrt(np.mean((A(u+warp(u))@m-p['v2'])**2)))
base=s['panel_b']['v']; r=inv(q,b); fw=lambda z:A(z)@b; v2=lambda z:A(z+warp(z))@m; x2={'baseline':inv(ib+v2(ib),b),'B-minus-even':inv(ib-warp(r)@m[1:],b),'inverse-B-reflected-quarter':inv(r-.25*warp(r),b),'inverse-B-reflected-half':inv(r-.5*warp(r),b),'B-native-reflected':inv(ib-warp(ib)@m[1:],b),'latent-reflected':q-warp(q),'B-reflected':ib-warp(ib),'B-pure-warp':ib+warp(ib)}
rng=np.random.default_rng(36521); splits=[np.arange(len(y))]+[rng.permutation(len(y)) for _ in range(30)]; splits += [np.r_[np.argsort(x[:,j])[20:],np.argsort(x[:,j])[:20]] for j in range(4)]
for name,w in x2.items():
 err=[]
 for ix in splits:
  a0,b0=ix[:52],ix[52:]; pred=f['_scalar']({'x':x[ix],'i2b':w[ix]},y[a0]); err.append(np.mean(np.abs(pred-y[b0])))
 print(name,'official, fresh_mean, fresh_worst, boundary_worst',np.round([err[0],np.mean(err[1:31]),max(err[1:31]),max(err[31:])],6))
