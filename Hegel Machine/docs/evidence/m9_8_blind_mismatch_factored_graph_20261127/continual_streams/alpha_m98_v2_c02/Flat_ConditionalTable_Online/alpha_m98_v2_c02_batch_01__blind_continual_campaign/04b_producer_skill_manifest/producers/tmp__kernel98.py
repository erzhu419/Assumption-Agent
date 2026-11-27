exec(open('/tmp/law98.py').read().split('for mode in range')[0])
cal=d['calibration_load_cells']; raw=cal['bridge_raw']; coef=np.linalg.lstsq(np.c_[raw,np.ones(len(raw))],cal['true_channel_loads'],rcond=None)[0]
cv=lambda r:r@coef[:4]+coef[4]
S=cv(a['past_channel_raw']); H=cv(a['hold_channel_raw']); print('stress and channel std:',np.std(S,axis=0)); print('channel example',S[:3].round(3)); print('past vs hold',np.c_[s,h].round(3)); print('feature sensitivity:')
for nm,(l,r) in {'train_diag':(d['train'],a),'all_diag':(d['train'],xx),'test_diag':(d['test'],a),'diag_only':(a,a)}.items(): print(nm,np.c_[np.ones(len(h)),m['_standardize'](l,r)[1][-len(h):]]@b[2:]);
# Force the known hold contribution; model each candidate history correction as a shared scalar response.
from scipy.optimize import minimize_scalar
ids=np.arange(len(h));
for label,X in [('past',s),('past+const',np.c_[np.ones(len(h)),s]),('channels at last',S[:,-1,:]),('channel average',S.mean(1)),('decayed channels fast',np.einsum('nsc,s->nc',S,W[:,0])),('decayed channels slow',np.einsum('nsc,s->nc',S,W[:,1])),('past+hold channels',np.c_[s,H]),('history simple',np.c_[s[:,0],s[:,-2]-s[:,-1],s.mean(1)]),('past+nonlinear drift',np.c_[s,q*q,q**3]),('past+lab',np.c_[s,np.eye(4)[a['lab_id']]])]:
 def solve(ii):
  def model(p,ret=False):
   v=expb(p[0]); jv=expb(p[1]); G=h[:,None]*(b[:2]@T)+p[3]*q[:,None]*v
   Z=X[:,None,:]*(T[0]+p[2]*T[1])[None,:,None]
   if label=='past+nonlinear drift': Z[:,:,6:]=X[:,None,6:]*jv[None,:,None]
   C=np.linalg.lstsq(Z[ii].reshape(-1,X.shape[1]),(Y-G)[ii].ravel(),rcond=None)[0]; E=Y-G-Z@C
   return (E,C) if ret else E[ii].ravel()
  p=least_squares(model,[2.2,2.2,5,1.7],bounds=([.15,.15,-40,-10],[20,20,40,10]),max_nfev=70).x; return (*model(p,True),p)
 E,C,p=solve(ids); loo=np.stack([solve(ids[ids!=i])[0][i] for i in ids]); print(label,'p',p,'coeff',C,'in',rm(E),'LOO',rm(loo)/[1,1,1,.7]);
