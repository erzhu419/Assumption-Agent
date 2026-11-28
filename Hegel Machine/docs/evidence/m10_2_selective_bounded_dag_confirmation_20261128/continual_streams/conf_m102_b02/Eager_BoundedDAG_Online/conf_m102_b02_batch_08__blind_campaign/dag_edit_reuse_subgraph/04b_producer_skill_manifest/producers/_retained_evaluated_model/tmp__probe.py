import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def D(x):return np.c_[np.ones(len(x)),x]
def A(u):return D(np.c_[u,u*np.roll(u,-1,axis=1),u**3])
def fit(x,y):return np.linalg.lstsq(x,y,rcond=1e-10)[0]
a,b,c=(s['panel_'+k] for k in 'abc'); ac=fit(A(c['u']),c['v1']); print('C1 coefficients',ac.round(4)); print('C1 on A residuals',np.sort(np.linalg.norm(A(a['u'])@ac-a['v'],axis=1)).round(4))
F=A(a['u']); v=a['v']; rng=np.random.default_rng(8)
w=min((fit(F[i],v[i]) for i in [rng.choice(len(F),15,replace=False) for _ in range(900)]),key=lambda w: np.median(np.linalg.norm(F@w-v,axis=1))); r=np.linalg.norm(F@w-v,axis=1); print('A robust residual',np.sort(r).round(4)); print('A coefficients',fit(F[r<max(.002,4*np.median(r))],v[r<max(.002,4*np.median(r))]).round(4))
forms={'dead':(lambda u,q:np.sign(u)*np.maximum(abs(u)-q[0],0)**q[1],[.3,1.5],([0,.25],[.9,4])), 'tanh':(lambda u,q:np.tanh(q[0]*u),[1],([.001],[10])), 'sin':(lambda u,q:np.sin(q[0]*u),[1],([.001],[5])), 'rat':(lambda u,q:u/(1+q[0]*u*u),[.5],([0],[20])), 'asinh':(lambda u,q:np.arcsinh(q[0]*u),[1],([.001],[15]))}
for p,k in [(b,'v'),(c,'v2')]:
 u,y=p['u'],p[k]; print('PANEL',k)
 for n,(fn,q,bds) in forms.items():
  def err(q):
   h=D(fn(u,q));return (h@fit(h,y)-y).ravel()
  opt=least_squares(err,q,bounds=bds); print(n,opt.x,np.sqrt(np.mean(opt.fun**2))); 
  if n=='dead': print(fit(D(fn(u,opt.x)),y).round(4))
