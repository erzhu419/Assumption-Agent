import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task');import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];u,v=[s['panel_b'][k] for k in ('u','v')];ls=m._ls
rows=[]
for tag,fun in [('tanh3',lambda p,g:np.tanh(g*p**3)),('atan3',lambda p,g:np.arctan(g*p**3)),('rat3',lambda p,g:p**3/(1+g*p**6)),('hill',lambda p,g:p**3/np.sqrt(1+g*p**6)),('tanhpow',lambda p,g:np.tanh(g[0]*np.sign(p)*abs(p)**g[1])),('atanpow',lambda p,g:np.arctan(g[0]*np.sign(p)*abs(p)**g[1]))]:
 for shift in [0,1]:
  for extra in [(),('raw',),('cyc',),('raw','cyc'),('squ',),('raw','squ'),('raw','aaa')]:
   def feat(x,g):
    p=x+(g[-4:] if shift else 0); q=g[:2] if 'pow' in tag else g[0]; b=fun(p,q); d={'raw':x,'squ':x*x,'cyc':x*m._roll(x),'aaa':x**3};return m._aff(np.c_[b,*[d[k] for k in extra]])
   def err(g):
    a=feat(u,g); return (a@ls(a,v)-v).ravel()
   gs=([1.25,3.] if 'pow' in tag else [1.25]); nn=len(gs);opt=least_squares(err,gs+[0.]*4*shift,bounds=([.03]*nn+[-.4]*4*shift,[8.]*nn+[.4]*4*shift),max_nfev=80); a=feat(u,opt.x);h=(a*np.linalg.pinv(a).T).sum(1); e=np.mean(abs(opt.fun.reshape(-1,4)/(1-h)[:,None]));rows.append((e,np.std(opt.fun),tag,shift,extra,opt.x))
for r in sorted(rows)[:12]:print(r)
# Inspect panel input support and cross-domain test geometry.
for k in ['panel_a','panel_b','panel_c']:
 p=s[k]; print(k,p['u'].min(0),p['u'].max(0))
