from paths import *
np.set_printoptions(precision=5,suppress=True,linewidth=150)
z=cal(T['x']); u,v=(S['panel_b'][k] for k in ('u','v')); yt=T['y']; tr=T['traj']; t=S['eval_times']; U,s,V=np.linalg.svd(tr,full_matrices=False); B=V[0]/V[0,-1]; amp=tr@B/(B@B)
print('time',B,'second',V[1]);
for k,P in [('exp',lambda a:-np.expm1(-t*a)),('log',lambda a:np.log1p(t*a)),('root',lambda a:np.sqrt(1+t*a)-1),('rational',lambda a:t/(1+a*t)),('asinh',lambda a:np.arcsinh(t*a))]:
 fit=least_squares(lambda a:P(a)/P(a)[-1]-B,.9,bounds=(.001,30)); print(k,fit.x,np.linalg.norm(fit.fun),np.linalg.norm(P(1.25)/P(1.25)[-1]-B))
extra={'erf':lambda x,p:__import__('scipy').special.erf(p*x),'softclip':lambda x,p:x/(1+abs(p*x)),'genexp':lambda x,p:x*np.exp(-p[0]*abs(x)**p[1]),'gaussshift':lambda x,p:x*np.exp(-p[0]*(x-p[1])**2)}; funcs={**nl,**extra};
for name,h in funcs.items():
 for variant in ('scalar','separate','joint') if name not in ('genexp','gaussshift') else ('scalar',):
  def F(x,p):return np.c_[af(x),h(x,p)]
  def fun(p):
   f=F(u,p); e=(f@ls(f,v)-v).ravel();
   if variant=='joint':
    a=F(z[:52],p); e=np.r_[e,(a@ls(a,amp[:52])-amp[:52])*.75]
   return e
  if name=='genexp': p0,limits=[1.,2.],([.05,.6],[5,5]);
  elif name=='gaussshift': p0,limits=[1.,0.],([.05,-.9],[5,.9]);
  else: p0,limits=np.ones(1 if variant=='scalar' else 4),(.05,5)
  p=least_squares(fun,p0,bounds=limits,gtol=1e-11).x; f=F(u,p); c=ls(f,v); g=F(z,p)@c; H=np.linalg.qr(f)[0]; e=(f@c-v)/(1-(H*H).sum(1))[:,None]; P=quad(g); coeff=ls(P[:52],yt[:52]); Q=quad(g,True); states=af(g); print(name,variant,p,'panel',np.sqrt((e*e).mean()),'state',abs(states[52:]@ls(states[:52],tr[:52])-tr[52:]).mean(),'y_cross',abs(Q[52:]@ls(Q[:52],yt[:52])-yt[52:]).mean(),'y_full',abs(P[52:]@coeff-yt[52:]).mean(), 'coef', coeff if name=='gauss' else '')
  if name=='gauss' and variant=='scalar':
   print('panel coefficients',c,'state coef',ls(states[:52],amp[:52]),'y linear',ls(states[:52],yt[:52])); print('y residual corr with amplitude',np.corrcoef(amp,yt-states@ls(states,yt))[0,1]); print('amp mismatch',np.quantile(amp-states@ls(states,amp),[0,.1,.5,.9,1])); print('h range',g.min(0),g.max(0)); print('quad sv',np.linalg.svd(P[:52],compute_uv=False))
