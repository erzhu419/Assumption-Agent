exec(open('/tmp/search.py').read().split('for basis in')[0])
from scipy.optimize import least_squares
rng=np.random.default_rng(1)
for mode in ['sin','cos','tanh','rational','singate','ratgate','sincos']:
 def feat(w,vals):
  q=vals@w[:4]+w[4]; f={'sin':np.sin(q),'cos':np.cos(q),'tanh':np.tanh(q),'rational':q/(1+q*q),'singate':np.sin(q),'ratgate':1/(1+q*q),'sincos':np.sin(q)}[mode]
  if mode in ['singate','ratgate']: return a(vals)*f[:,None]
  if mode=='sincos':return np.c_[f,np.cos(q)]
  return f[:,None]
 for X in [base,Q]:
  i,j=splits[0]; xi=X[i]; pin=np.linalg.pinv(xi); rr=y[i]-xi@(pin@y[i]);
  def residual(w):
   f=feat(w,z[i]); f-=xi@(pin@f); return rr-f@fit(f,rr)
  candidates=[least_squares(residual,np.r_[v,0.1],max_nfev=80) for v in np.r_[np.eye(4)*1.7,rng.normal(0,1.3,(24,4))]]
  opt=min(candidates,key=lambda v:v.cost); F=np.c_[X,feat(opt.x,z)]; print(mode,X.shape[1],opt.x,opt.cost,score(F),fit(F[i],y[i])[-(F.shape[1]-X.shape[1]):])
