exec(open('/tmp/probe.py').read().split("print('times'")[0])
from scipy.optimize import least_squares
z=(x-a[0])/a[1]; w=A(z)@b
p=s['panel_c']; u=p['u'];
for k in ('v1','v2'):
 c=fit(P(u,2),p[k]); print(k,'quad',c.T, 'train residual max',np.max(abs(P(u,2)@c-p[k])))
def calc(f):
 h=f@fit(f[:n],y[:n]); return tuple(np.mean(abs(i)) for i in ((h-y)[:n],(h-y)[n:]))
c=fit(A(z)[:n],y[:n]); d=fit(A(z)[:n],t[:n,-1]); print('y and state linear',c,d); print('index poly',[(i,calc(np.c_[A(z),*[ (z@c[1:])**j for j in range(2,i+1)]])) for i in range(2,8)])
for coords,q in zip(['z','w','x','C1z','C2z','C1w','C2w'],[z,w,x,P(z,2)@fit(P(u,2),p['v1']),P(z,2)@fit(P(u,2),p['v2']),P(w,2)@fit(P(u,2),p['v1']),P(w,2)@fit(P(u,2),p['v2'])]):
 print(coords,'linear/quadratic',calc(A(q)),calc(P(q,2)),'append',calc(np.c_[A(z),q]))
 for i in range(4):
  print(i,calc(np.c_[A(z),q[:,i]**2,q[:,i]**3]),calc(np.c_[A(z),q[:,i],q[:,i]**2,q[:,i]**3,q[:,i]**4]))
for use_linear in (True,False):
 for mode in ('poly3','poly5','tanh','sin','rational'):
  start=c[1:]/np.linalg.norm(c[1:]); base=A(z) if use_linear else np.ones((len(z),1));
  def feats(h):
   g=z@h[:4]; j=g+h[4];
   if mode=='poly3': f=np.c_[g,g*g,g**3]
   elif mode=='poly5': f=np.c_[g,g*g,g**3,g**4,g**5]
   elif mode=='tanh': f=np.tanh(j)[:,None]
   elif mode=='sin': f=np.sin(j)[:,None]
   else: f=(j/(1+j*j))[:,None]
   return np.c_[base,f]
  def err(h):
   f=feats(h)[:n]; return f@fit(f,y[:n])-y[:n]
  rng=np.random.default_rng(93); sols=[least_squares(err,h,max_nfev=150) for h in [np.r_[start*k,0.] for k in (.6,1.5,3)] + [np.r_[rng.normal(size=4),0.] for _ in range(7)]]; best=min(sols,key=lambda o:o.cost); print('index',use_linear,mode,calc(feats(best.x)),best.x)
