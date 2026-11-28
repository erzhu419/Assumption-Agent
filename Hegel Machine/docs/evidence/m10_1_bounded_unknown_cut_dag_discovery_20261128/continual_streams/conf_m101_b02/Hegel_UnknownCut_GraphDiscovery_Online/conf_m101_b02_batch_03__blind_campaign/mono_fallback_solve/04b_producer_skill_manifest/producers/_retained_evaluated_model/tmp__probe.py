exec(open('/tmp/scan.py').read().split("p=s['panel_a'];")[0])
from scipy.optimize import minimize_scalar
np.set_printoptions(precision=5,suppress=True)
p=s['panel_a'];u,v=p['u'],p['v']; a=design(u); mask=np.ones(len(u),bool)
for _ in range(3):
 b=ls(a[mask],v[mask]); r=np.linalg.norm(a@b-v,axis=1); mask=np.argsort(r)[:45]
u,v=u[mask],v[mask]; print('A trimmed affine',b,'resid',np.sort(r)[:45]);print('A u->v ranges',u.min(0),u.max(0),v.min(0),v.max(0)); print('first pairs',np.c_[u,v][:8]);
families={'sq':lambda u,a:u*u,'cubic':lambda u,a:u**3,'tanh':lambda u,a:np.tanh(a*u),'sin':lambda u,a:np.sin(a*u),'cos':lambda u,a:np.cos(a*u),'oddexp':lambda u,a:-np.sign(u)*np.expm1(-a*abs(u)),'rational':lambda u,a:u/(1+a*u*u),'softsq':lambda u,a:u*u/np.sqrt(1+a*u*u)}
for direction, U,V in [('forward',u,v),('inverse',v,u)]:
 for name,func in families.items():
  def fun(q):
   a=design(U,func(U,q)); return np.linalg.norm(a@ls(a,V)-V)/np.sqrt(len(U))
  opt=minimize_scalar(fun,method='bounded',bounds=(.005,5));print(direction,name,opt.fun,opt.x)
for k in ('panel_b',):
 p=s[k];a=design(p['u']);b=ls(a,p['v']);print(k,b, np.linalg.svd(b[1:])[1]);
x=s['train']['x'];y=s['train']['y'];t=s['train']['traj'];print('times',s['eval_times'],'tel',s['telemetry_vec']);print('x covariance',np.cov(x.T));print('test covariance',np.cov(s['test']['x'].T));
_,_,vt=np.linalg.svd(t,full_matrices=False);print('trajectory basis',vt[:2]);target=np.c_[y,t@vt[:2].T];a=design(x); w=ls(a[:52],target[:52]); print('linear target coeff',w); rr=target-a@w; print('linear MAE',np.mean(abs(rr[52:]),axis=0));print('targets vs linear fitted',np.c_[target, a@w][:15]);
