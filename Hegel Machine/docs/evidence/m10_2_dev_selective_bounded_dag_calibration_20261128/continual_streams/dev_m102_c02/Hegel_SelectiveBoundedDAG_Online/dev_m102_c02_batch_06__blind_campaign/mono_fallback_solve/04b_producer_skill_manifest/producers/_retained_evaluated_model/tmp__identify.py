from graph import *
a,inv=Amap(s['panel_a']); raw=s['train']['x']; x=inv(raw); y=s['train']['y']; t=s['eval_times']; tr=s['train']['traj']; print('times',t,'svd',np.linalg.svd(tr,compute_uv=False));print('X corrected range',x.min(0),x.max(0));print('tr',tr[:12]); print('y',y[:12]); print('t diff ratios', (np.diff(tr,axis=1)/np.diff(t))[:12]);
z=np.c_[y,tr]; rng=np.random.default_rng(7); splits=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in [rng.permutation(72) for _ in range(3)]]
def scores(P,alpha=.01):
 out=[]
 for i,j in splits:
  m=P[i].mean(0); sd=P[i].std(0).clip(1e-10); X=(P[i]-m)/sd; Y=(P[j]-m)/sd; E=np.abs(Y@np.linalg.solve(X.T@X+alpha*np.eye(X.shape[1]),X.T@z[i])+z[i].mean(0)-z[j]); out.append([E[:,0].mean(),E[:,1:].mean()]);
 return np.array(out)
for label,x in [('raw',raw),('inv',x)]:
 print(label)
 X=poly(x,1); Q=poly(x,2)
 odd={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'asinh':np.arcsinh,'rat':lambda u:u/(1+u*u),'sat':lambda u:u/np.sqrt(1+u*u)}
 R=[]
 for nm,f in odd.items():
  for scale in [.4,.7,1,1.4,2,3,4]:
   for base,P in [('linear',X),('quad',Q),('diag',np.c_[X,x*x])]:
    for uv,w in [('coord',x),('sum',np.c_[x,*[(x[:,i]+sg*x[:,j]) for i in range(4) for j in range(i) for sg in [-1,1]]])]:
     for alpha in [.002,.1,1]:
      v=scores(np.c_[P,f(w*scale)],alpha);R.append((np.mean(v,0),v[0],nm,scale,base,uv,alpha))
 for k in range(2):
  print(sorted(R,key=lambda o:o[0][k])[:10]);
 # nonlinear index fits with variable projection, with or without pairwise terms
 for label,P in [('lin',X),('quad',Q)]:
  for tar in [0,1,8]:
   for Fn in [np.tanh,np.sin,lambda q:q/(1+q*q),lambda q:np.exp(-q*q),lambda q:q*q/(1+q*q)]:
    target=z[:,tar]; best=(1e9,)
    for _ in range(15):
     def make(p):return np.c_[P,Fn(X@p)]
     def fun(p):
      W=make(p)[:52]; return W@fit(W,target[:52])-target[:52]
     opt=least_squares(fun,rng.normal(0,.6,5),max_nfev=120); W=make(opt.x); err=np.mean(abs(opt.fun)); pred=W@fit(W[:52],target[:52]);
     if err<best[0]:best=(err,np.mean(abs(pred[52:]-target[52:])),opt.x)
    print('ridge',label,tar,Fn.__name__,best)
