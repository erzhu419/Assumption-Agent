exec(open('/tmp/explore.py').read().split('a=s[')[0])
from itertools import combinations,product
u=s['panel_c']['u']; v=s['panel_c']['v2']; ii,jj=np.triu_indices(4,1)
W={'u':u, 'pairprod':u[:,ii]*u[:,jj], 'cross':u[:,[0,0,1,2]]*u[:,[1,3,2,3]], 'square':u*u, 'sum':u[:,ii]+u[:,jj], 'diff':u[:,ii]-u[:,jj]}
best=[]
for name,w in W.items():
 for bn,b in [('const',np.ones((len(u),1))),('lin',ones(u)),('cross',ones(W['cross'])),('lin+cross',ones(np.c_[u,W['cross']]))]:
  for fun in ['sin','cos','tanh','softsign','rational','trig']:
   def f(p):
    z=w*p[0]+p[1:] if fun!='trig' else w*p[0];
    if fun=='trig':a=np.c_[np.sin(z),np.cos(z)]
    elif fun=='rational':a=1/(1+z*z)
    elif fun=='softsign':a=z/(1+abs(z))
    else:a=getattr(np,fun)(z)
    return np.c_[b,a]
   def err(p):
    F=f(p); return (F@ls(F,v)-v).ravel()
   for st in [0.75,1.25,2.5,4.]:
    p=least_squares(err,[st]+([0.01]*w.shape[1] if fun!='trig' else []),max_nfev=90).x; F=f(p); H=np.linalg.qr(F)[0]; e=err(p).reshape(v.shape); cv=np.mean(abs(e)/np.maximum(.01,1-(H**2).sum(1,keepdims=True)));best.append((cv,np.mean(abs(e)),name,bn,fun,p))
print(*sorted(best,key=lambda a:a[0])[:24],sep='\n')
t=s['eval_times']; traj=s['train']['traj']; T=traj[:52].T
out=[]
for name,a in [('exp',lambda z:-np.expm1(-z)),('dexp',lambda z:z*np.exp(-z)),('sine',np.sin),('cosine',lambda z:1-np.cos(z)),('linear',lambda z:z),('square',lambda z:z*z),('rise2',lambda z:(-np.expm1(-z))**2),('gate',lambda z:z*(-np.expm1(-z)))]:
 for nfix in [0,1,2]:
  def F(p):return np.c_[-np.expm1(-t*p[0]),-np.expm1(-t*p[1]),a(t*p[2])]
  for st in [[2.,.22,1.25],[1.25,.22,2.5],[.6,2.5,1.25]]:
   def pars(q):return np.r_[st[:nfix],q]
   def res(q):
    h=F(pars(q));return (h@ls(h,T)-T).ravel()
   p=pars(least_squares(res,st[nfix:],bounds=(.005,15),max_nfev=150).x);h=F(p); pr=(h@ls(h,traj[52:].T)).T; out.append((np.mean(abs(pr-traj[52:])),np.mean(abs(res(p[nfix:]))),name,p))
print('time modes',*sorted(out,key=lambda a:a[1])[:22],sep='\n')
