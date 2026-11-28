exec(open('/tmp/probe.py').read().split("print('telemetry'")[0])
def cyclic(x):return np.roll(x,-1,axis=1)
def aop(s):
 u,v=s['panel_a']['u'],s['panel_a']['v']; _,_,m=clip(af(u),v)
 def d(x,j):
  h,b=x[:,j],x[:,(j+1)%4];return np.c_[np.ones(len(h)),h,b,h*b,h**3]
 w=[ls(d(u[m],j),v[m,j]) for j in range(4)]
 return lambda x:np.column_stack([d(x,j)@w[j] for j in range(4)])
def ops(s):
 b=s['panel_b']; c=s['panel_c']; u=c['u']; w=ls(af(b['u']),b['v'])
 def f(x):return af(x*cyclic(x))
 w1=ls(f(u),c['v1']);
 def g(x,k):return af(np.cos(k*x))
 def r(k):
  a=g(u,k);return (a@ls(a,c['v2'])-c['v2']).ravel()
 k=least_squares(r,[2.4]).x;w2=ls(g(u,k),c['v2']);return (lambda x:af(x)@w,lambda x:f(x)@w1,lambda x:g(x,k)@w2)
x=s['train']['x'];a=aop(s)(x);B,C,D=ops(s)
blocks={'x':x,'a':a,'b':B(x),'ba':B(a)}
for k,z in list(blocks.items()):blocks['c'+k]=C(z);blocks['d'+k]=D(z)
y=s['train']['y'];tr=s['train']['traj'];
def trial(f,t):
 inds=np.array_split(np.random.default_rng(31).permutation(len(y)),4)+[]; inds.append(np.arange(52,72));
 return np.array([np.mean(abs(f[te]@ls(f[np.setdiff1d(np.arange(len(y)),te)],t[np.setdiff1d(np.arange(len(y)),te)])-t[te])) for te in inds])
def designs(p,r):
 pr=(p[:,:,None]*r[:,None,:]).reshape(len(p),-1); yield 'lin',af(np.c_[p,r]); yield 'dot',af(np.c_[p,r,np.sum(p*r,axis=1)]);yield 'pair',af(np.c_[p,r,pr]);yield 'diag',af(np.c_[p,r,p*r]);
for t in [y,tr]:
 res=[]
 for (pn,p),(rn,r) in itertools.combinations(blocks.items(),2):
  for mode,f in designs(p,r):
   z=trial(f,t);res.append((z.mean(),z.max(),z[-1],pn,rn,mode))
 print('TOP',t.shape);print(*sorted(res)[:18],sep='\n')
