exec(open('/tmp/probe.py').read().split('print(\'A robust')[0])
from scipy.optimize import least_squares
p=s['panel_b'];cb=fit(lin(p['u']),p['v']); p=s['panel_c']; u=p['u']; U=lin(u)
np.set_printoptions(precision=4,suppress=True)
print('B gram',cb[1:].T@cb[1:]); print('C sample',np.c_[u,p['v1'],p['v2']][:8]); print('C output extrema',[(p[k].min(0),p[k].max(0)) for k in ['v1','v2']]); print('C u extrema',u.min(0),u.max(0))
acts={'sin':np.sin,'tanh':np.tanh,'atan':np.arctan,'softsign':lambda x:x/(1+np.abs(x)),'rat':lambda x:x/(1+x*x),'cos':np.cos,'sig':lambda x:1/(1+np.exp(np.clip(-x,-100,100))),'cubic':lambda x:x+x**3,'sat':lambda x:x/(1+x*x)**.5,'exp':lambda x:np.exp(np.clip(x,-10,10))}
for k in ('v1','v2'):
 for j,v in enumerate(p[k].T):
  b=fit(U,v); o=[]
  for name,f in acts.items():
   def fun(z):return z[5]+z[6]*f(U@z[:5])-v
   opts=[least_squares(fun,np.r_[b*c,0.,1.],max_nfev=120) for c in [.5,1,2,3]]
   r=min(opts,key=lambda z:np.linalg.norm(z.fun));o.append((np.linalg.norm(r.fun)/len(u)**.5,name,r.x.round(4)))
  print(k,j,sorted(o,key=lambda a:a[0])[:4]); print('quad residual',np.std(v-np.c_[U,poly(u,2)]@fit(np.c_[U,poly(u,2)],v)))
