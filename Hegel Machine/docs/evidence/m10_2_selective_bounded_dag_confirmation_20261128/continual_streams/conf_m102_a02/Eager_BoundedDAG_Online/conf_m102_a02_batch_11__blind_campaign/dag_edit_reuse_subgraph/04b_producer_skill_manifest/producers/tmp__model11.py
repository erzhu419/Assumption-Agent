exec(open('/tmp/probe11.py').read().split('for k,p in')[0])
from scipy.optimize import least_squares
F={'erf':lambda z,w:erf(w*z),'tanh':lambda z,w:np.tanh(w*z),'sin':lambda z,w:np.sin(w*z),'rat':lambda z,w:z/(1+w*z*z),'soft':lambda z,w:z/np.sqrt(1+w*z*z),'ash':lambda z,w:np.arcsinh(w*z),'atan':lambda z,w:np.arctan(w*z)}
p=s['panel_a'];u=p['u'];v=p['v'];m=np.max(abs(v),axis=1)<3;u=u[m];v=v[m]; xx=np.r_[x,s['test']['x']]
def cal(name,w,aug=None):
 f=F[name]; a=lambda u:np.c_[aff(u),f(u,w),*([] if aug is None else [F[aug[0]](u,aug[1])])]; c=fit(a(u),v); lin=fit(aff(u),v); z=np.array([least_squares(lambda zz:a(zz[None])[0]@c-b, np.linalg.solve(lin[1:].T,b-lin[0]),max_nfev=40).x for b in xx]);return z,np.sqrt(np.mean((a(u)@c-v)**2)),c
p=s['panel_c'];h=p['u'];q=p['v1'];r=p['v2'];
def c2(h,kind,w):
 g={'rat':lambda t:t*t/(1+w*t*t),'cos':lambda t:1-np.cos(w*t),'exp':lambda t:-np.expm1(-w*t*t),'sqrt':lambda t:np.sqrt(1+w*t*t)-1,'log':lambda t:np.log1p(w*t*t),'tanh':lambda t:np.tanh(w*t)**2,'mix':lambda t:np.tanh(w*t)*t}[kind]; return np.c_[aff(h),g(h)]
opts=[]
for kind in ['rat','cos','exp','sqrt','log','tanh','mix']:
 def loss(w):
  a=c2(h,kind,w);return np.mean((a@fit(a,r)-r)**2)
 o=minimize_scalar(loss,bounds=(.001,3.5),method='bounded');opts.append((o.fun,kind,o.x)); print('C2',kind,o.x,np.sqrt(o.fun))
_,kind,w2=min(opts);print('C2 coefficient',fit(c2(h,kind,w2),r).round(4))
# Joint odd/even panel fits, including shared scale and simulated folds.
xs=[]
for name,w in [('erf',1.17143),('tanh',1.12409),('sin',2.43856),('ash',1.25)]:
 z,e,c=cal(name,w);xs.append((name,z));print('A',name,e,c.round(4),'range',z.min(),z.max())
 amp=t@np.linalg.svd(t,full_matrices=False)[2][0];print('amp',amp.min(),amp.max());
 for lab,f,w in [('tanh',F['tanh'],2.15),('ash',F['ash'],3.38923),('erf',F['erf'],2.15),('ashT',F['ash'],2.15),('atan',F['atan'],2.15)]:
  phi=lambda a:np.c_[aff(a),f(a,w)];g=phi(z)@fit(phi(h),q);b=c2(z,kind,w2)@fit(c2(h,kind,w2),r)
  a0=aff(z); d={'z':a0,'c1':aff(g),'c2':aff(b),'c1+c2':np.c_[a0,g,b],'z+c1':np.c_[a0,g], 'z+c2':np.c_[a0,b], 'poly':polys(z,2),'poly+panels':np.c_[polys(z,2),g,b]}
  for op in F:
   for ww in (1.25,2.15):d[op+str(ww)]=np.c_[a0,F[op](z,ww),b]
  ysc=[];tsc=[]
  for dn,a in d.items():
   a=a[:len(x)];ysc.append((np.mean(abs(a[n:]@fit(a[:n],y[:n])-y[n:])),dn));tsc.append((np.mean(abs(a[n:]@fit(a[:n],t[:n])-t[n:])),dn))
  print(name,lab,'y',sorted(ysc)[:3],'t',sorted(tsc)[:3])
np.savez('/tmp/probe11_latent.npz',**dict(xs))
