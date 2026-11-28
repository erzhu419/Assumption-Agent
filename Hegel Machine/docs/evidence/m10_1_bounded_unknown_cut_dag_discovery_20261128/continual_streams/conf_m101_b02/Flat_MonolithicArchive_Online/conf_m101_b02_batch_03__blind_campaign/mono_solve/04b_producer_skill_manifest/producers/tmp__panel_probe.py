import pickle,numpy as np
from scipy.optimize import minimize_scalar, least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def D(x): return np.c_[np.ones(len(x)),x]
def L(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
def fit(g,y):
 c=L(g,y);return np.sqrt(np.mean((y-g@c)**2)),c
p=s['panel_a']; x,y=p['u'],p['v']; e=np.linalg.norm(y-D(x)@L(D(x),y),axis=1); k=np.argsort(e)[:45]; x,y=x[k],y[k]
for n in (1,2,3,4):
 cols=[a for j in range(1,n+1) for a in combinations_with_replacement(range(4),j)]; g=D(np.array([x[:,a].prod(1) for a in cols]).T); err,c=fit(g,y); ie=np.arange(len(g))%5==0; cv=(np.mean((g[ie]@L(g[~ie],y[~ie])-y[ie])**2))**.5
 print('A polynomial',n,err,cv,[(a,np.round(b,3).tolist()) for a,b in zip(cols,c[1:]) if abs(b).max()>.055])
forms={'odd':lambda x,r:-np.sign(x)*np.expm1(-r*abs(x)),'even':lambda x,r:x*x/np.sqrt(1+r*x*x),'sin':lambda x,r:np.sin(r*x),'tanh':lambda x,r:np.tanh(r*x),'cos':lambda x,r:np.cos(r*x),'rational':lambda x,r:x/(1+r*x*x)}
for k,j in [('panel_a','v'),('panel_c','v1'),('panel_c','v2')]:
 p=s[k]; u,v=(x,y) if k=='panel_a' else (p['u'],p[j]); out=[]
 for name,f in forms.items():
  for usex in (False,True):
   g=lambda r:D(np.c_[u,f(u,r)] if usex else f(u,r)); o=minimize_scalar(lambda r:fit(g(r),v)[0],bounds=(.05,4),method='bounded'); out.append((o.fun,name,usex,o.x))
 print(k,j,sorted(out)[:6])
z=s['train']['traj']; t=s['eval_times'];
for typ in ('two_exp','exp_and_t','exp_and_bias','exp_and_exp_t'):
 def g(r):
  a=-np.expm1(-r[0]*t); b={'two_exp':-np.expm1(-r[-1]*t),'exp_and_t':t,'exp_and_bias':np.ones(len(t)),'exp_and_exp_t':t*np.exp(-r[0]*t)}[typ];return np.array([a,b]).T
 def res(r):
  u=g(r);return (u@L(u,z.T)-z.T).ravel()
 r=least_squares(res,[.7,1.7] if typ=='two_exp' else [.7],bounds=(.001,8));print(typ,r.x,np.sqrt(np.mean(r.fun**2)), np.linalg.svd(g(r.x),compute_uv=False))
