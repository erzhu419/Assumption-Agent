exec(open('/tmp/probe08.py').read().split('for f in sorted')[0]); m=load(glob.glob('/tmp/hegel_continual_archive/*06_solver.py')[0]); q=m._invert(x,m._fit_calibration(s['panel_a'])); u=s['panel_b']['u'];v=s['panel_b']['v'];
def report(f):
 a=ls(f[:52],y[:52]);r=y-f@a;return np.round([err(r[:52]),err(r[52:]),err((f@ls(f,z)-z)).mean()],5)
for name,fn,ini,bounds in [('power',lambda a,p:a*np.expm1(p[2]*np.logaddexp(0,p[1]*np.log(np.maximum(abs(a)*p[0],1e-50)))),[2.5,12,.036],([.1,1,.001],[10,40,2])),('tanh',lambda a,p:np.tanh(p[0]*a),[2],([.1],[20.])),('asinh',lambda a,p:np.arcsinh(p[0]*a),[2],([.1],[20.])),('saturating',lambda a,p:a*np.exp(-p[2]*np.logaddexp(0,p[1]*np.log(np.maximum(abs(a)*p[0],1e-50)))),[2,4,.5],([.1,1,.001],[10,40,2]))]:
 def design(a,p):return np.c_[poly(a,1),fn(a,p)]
 def residual(p):
  f=design(u,p);return (f@ls(f,v)-v).ravel()
 sol=least_squares(residual,ini,bounds=bounds,max_nfev=2000,gtol=1e-13,ftol=1e-13,xtol=1e-13);f=design(u,sol.x);w=ls(f,v);b=design(q,sol.x)@w;print(name,sol.x,err(sol.fun),'coef',w); print('compact',report(np.c_[poly(b,1),b*np.roll(q,-1,axis=1)]));
 if name=='power':bp=b
b=bp; cyc=b*np.roll(q,-1,axis=1);base=np.c_[poly(b,1),cyc];r=y-base@ls(base,y); print('base coefs',ls(base,y));print('z coefs',ls(base,z[:,-1]));print('resid vs yhat',np.c_[y,base@ls(base,y),r][np.argsort(abs(r))[-15:]]);
a=np.c_[q,b,cyc]; names=['q'+str(i) for i in range(4)]+['b'+str(i) for i in range(4)]+['c'+str(i) for i in range(4)];cand=[]
for d in range(1,4):
 for t in cwr(range(12),d):
  col=np.prod(a[:,t],axis=1);adj=col-base@ls(base,col);n=np.linalg.norm(adj);
  if n>1e-8:cand.append((abs(r@adj/n),'*'.join(names[i] for i in t),report(np.c_[base,col]),col))
for score,key,e,col in sorted(cand,key=lambda t:-t[0])[:25]:print(round(score,4),key,e)
for name,f in [('qb',poly(np.c_[q,b])),('bc',poly(np.c_[b,cyc])),('qbc',poly(a)),('q cubic',poly(q,3)),('b cubic',poly(b,3))]:print(name,report(f));
