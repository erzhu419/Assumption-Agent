import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from graph05 import *
U,V=s['panel_b']['u'],s['panel_b']['v']
def fitwarp(fn,st,bd):
 def loss(p):
  h=A(fn(U,p));return (h@F(h,V)-V).ravel()
 o=least_squares(loss,st,bounds=bd,gtol=1e-13,max_nfev=500);return o.x,np.std(o.fun),A(fn(z,o.x))@F(A(fn(U,o.x)),V)
warps=[('threshold',lambda x,p:np.sign(x)*np.maximum(abs(x)-p[0],0)**p[1],[.22,1.25],([0,.1],[.7,8])),('threshold-rat',lambda x,p:np.sign(x)*np.maximum(abs(x)-p[0],0)**p[1]/(1+p[2]*x*x),[.22,1.25,.01],([0,.1,0],[.7,8,10])),('pow-threshold',lambda x,p:np.sign(x)*np.maximum(abs(x)**p[2]-p[0]**p[2],0)**p[1],[.22,1.25,1],([0,.1,.1],[.7,8,8]))]
for name,f,st,bd in warps:
 p,e,d=fitwarp(f,st,bd); a=A(d); print(name,p,e, 'traj',np.mean(abs(a[52:72]@F(a[:52],yt[:52])-yt[52:])))
 if name=='threshold':D=d
print('d range',np.min(D,0),np.max(D,0),'z range',z.min(0),z.max(0))
d=D; a=A(d); p1=d[:,0]*d[:,2];p2=d[:,1]*d[:,3]; p=p1+p2; h=np.c_[a,p]; co=F(h[:72],y); r=y-h[:72]@co; print('base',co,score(h),'resid',r.std()); print('corr residual by fitted',np.corrcoef(np.c_[r,h[:72]@co,(h[:72]@co)**3,d[:72]**3,z[:72],z[:72]**3].T)[0]);
results=[]
for aa,bb in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
 for ss in [1,-1]:
  v=d[:,aa]+ss*d[:,bb]
  for n,f in [('cube',v**3),('rat',v**3/(1+.22*v*v)),('signed',v*abs(v)),('pair3',d[:,aa]*d[:,bb]*v)]:results.append((score(np.c_[h,f]),aa,bb,ss,n))
print('pair direction',sorted(results,key=lambda j:j[0][1])[:10]); cub=prod(d,list(combinations_with_replacement(range(4),3))); quad=prod(d,list(combinations_with_replacement(range(4),2)))
for tag,H in [('cubic',np.c_[h,cub]),('q+c',np.c_[a,quad,cub]),('cubic+z',np.c_[h,cub,z]),('z+c',np.c_[h,z,z**3]),('branches',np.c_[h,w['c1powered'],w['c2even']])]:
 for rr in [0,.01,.1,1]:print(tag,rr,score(H,rr))
# Single index nonlinear saturation, with the pair term before or after activation.
for wh in ['tanh','rat','cubic','asinh','signed','exp','sin','quad','ratpair','ratplus','ratboth']:
 for post in [False,True]:
  errs=[]; pars=[]
  for i,j in folds:
   def predict(q):
    l=a@q[:5]; k=q[5]; v=l+(0 if post else k*p); t=q[6]
    fun={'tanh':lambda: np.tanh(t*v)/t,'rat':lambda:v/(1+t*v*v),'cubic':lambda:v+t*v**3,'asinh':lambda:np.arcsinh(t*v)/t,'signed':lambda:v+t*v*abs(v),'exp':lambda:v*np.exp(-t*v*v),'sin':lambda:np.sin(t*v)/t,'quad':lambda:v+t*v*v,'ratpair':lambda:v/(1+t*p*p),'ratplus':lambda:v/(1+t*(d[:,0]+d[:,1])**2),'ratboth':lambda:v/(1+t*np.sum(d*d,axis=1))}[wh]()
    return fun+(k*p if post else 0)
   base=F(h[i],y[i]); lo=-1 if wh in ['cubic','signed','quad'] else .00001
   q=least_squares(lambda q:predict(q)[i]-y[i],np.r_[base,.05],bounds=([-np.inf]*6+[lo],[np.inf]*6+[4])).x; pars.append(q); errs.append(np.mean(abs(predict(q)[j]-y[j])))
  print('INDEX',wh,post,np.round([errs[0],np.mean(errs[1:]),np.max(errs[1:])],6),'p',np.median(pars,axis=0))
print('cubic standardized coefs',list(zip(list(combinations_with_replacement(range(4),3)),F(np.c_[h,cub][:52],y[:52])[6:])))
