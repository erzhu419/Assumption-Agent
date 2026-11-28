exec(open('/tmp/discover.py').read().split('forms=')[0])
from scipy.special import erf
from scipy.spatial.distance import cdist
from scipy.linalg import solve
u,v=s['panel_b']['u'],s['panel_b']['v']; Y=np.c_[s['train']['y'],s['train']['traj']]; t=s['eval_times']; print('TELEMETRY',s['telemetry_vec'],'TIMES',t)
shapes={'exp':lambda z,p,q:z*np.expm1(-p*abs(z)**q),'rat':lambda z,p,q:z*(1-1/(1+p*abs(z)**q)),'atan':lambda z,p,q:z*np.arctan(p*abs(z)**q),'tanh':lambda z,p,q:z*np.tanh(p*abs(z)**q),'erf':lambda z,p,q:z*erf(p*abs(z)**q),'cos':lambda z,p,q:z*(1-np.cos(p*abs(z)**q)),'pow':lambda z,p,q:z*abs(z)**q}
results=[]
for j,sh in shapes.items():
 for mode in ['plain','cyc','opposite','all','square','cycsum','cycdiff','global']:
  for base in [False,True]:
   def f(z,p,j=j,sh=sh,mode=mode,base=base):
    a=sh(z,*p); b=np.roll(z,-1,axis=1); D={'plain':[],'cyc':[z*b],'opposite':[z[:,:2]*z[:,2:]],'all':[z[:,ia]*z[:,ja]],'square':[z*z],'cycsum':[sh(z+b,*p)],'cycdiff':[sh(z-b,*p)],'global':[z*np.sum(z*z,axis=1)[:,None]]}[mode]; return np.column_stack([aff(z) if base else np.ones(len(z)),a,*D])
   def e(p):
    A=f(u,p); return (A@ls(A,v)-v).ravel()
   p=min([least_squares(e,[a,b],bounds=([.02,.3],[8,6]),max_nfev=70) for a,b in [(1,2),(1,1),(.5,3),(3,2)]],key=lambda r:r.cost).x; A=f(u,p); W=ls(A,v); R=A@W-v; g=1-np.einsum('ij,ji->i',A,np.linalg.pinv(A)); results.append((np.mean(abs(R)/g[:,None]),np.mean(abs(R)),j,mode,base,p,lambda z,f=f,p=p,W=W:f(z,p)@W))
results.sort(key=lambda a:a[0]);
for r in results[:12]:print('PANELB',r[:-1]);
U=s['panel_c']['u']; cf=lambda q:np.c_[aff(q),q*np.exp(-1.0223*abs(q)**.9585)]; df=lambda q:np.c_[aff(q),q*q,np.exp(-.7109*q*q)]; C=lambda q:cf(q)@ls(cf(U),s['panel_c']['v1']); D=lambda q:df(q)@ls(df(U),s['panel_c']['v2']);
B=results[0][-1]; h=B(x); ha=next(r[-1] for r in results if r[2:5]==('exp','plain',True))(x)
F={'x':x,'b':h,'ba':ha,'c':C(x),'d':D(x),'bc':B(C(x)),'bd':B(D(x)),'cb':C(h),'db':D(h)}; F['xc']=np.c_[x,C(x)]; F['xb']=np.c_[x,h];F['cd']=np.c_[C(x),D(x)];F['bcd']=np.c_[h,C(x),D(x)]
tr=np.arange(52); te=np.arange(52,72); rng=np.random.default_rng(13); splits=[(tr,te)]+[(p[:52],p[52:]) for p in [rng.permutation(72) for i in range(4)]]
def score(A, ridge=0, gram=False):
 K=A if gram else None; out=[]
 for tr,te in splits:
  if gram: P=K[np.ix_(te,tr)]@solve(K[np.ix_(tr,tr)]+ridge*np.eye(len(tr)),Y[tr],assume_a='pos')
  elif ridge:
   sd=np.maximum(np.linalg.norm(A[tr],axis=0),1e-10); Z=A/sd; P=Z[te]@solve(Z[tr].T@Z[tr]+ridge*np.eye(Z.shape[1]),Z[tr].T@Y[tr],assume_a='pos')
  else: P=A[te]@ls(A[tr],Y[tr]);
  E=abs(P-Y[te]); out.append([E[:,0].mean(),E[:,1:].mean()]);
 return np.array(out)
R=[]
for n,z in F.items():
 i2,j2=np.triu_indices(z.shape[1]); Q=np.c_[aff(z),z[:,i2]*z[:,j2]]; A=Q; V=z/np.maximum(z.std(0),.04); G=cdist(V,V,'sqeuclidean')/z.shape[1];
 for d in [0,2,3,4]:
  if d<2: continue
  if d>=3 and z.shape[1]>4:continue
  for form in ['full','squarefree','cyclic']:
   I=[ix for k in range(d+1) for ix in it.combinations_with_replacement(range(z.shape[1]),k) if k<=2 or form=='full' or (len(set(ix))==k if form=='squarefree' else len(set(ix))<3)]; A=np.column_stack([np.prod(z[:,ix],axis=1) for ix in I]);
   for lam in [0,.0002,.003,.02]: R.append((score(A,lam),n,'poly',d,form,lam))
 for g in [.05,.15,.4,1,2,5]:
  for k in ['rbf','matern','rq']:
   a=np.sqrt(g*G); K={'rbf':np.exp(-a*a),'matern':(1+a+a*a/3)*np.exp(-a),'rq':1/(1+a*a)}[k]+1.;
   for lam in [1e-5,.0003,.003,.03]:R.append((score(K,lam,True),n,k,g,lam))
for col in (0,1):
 for k,f in [('fixed',lambda r:r[0][0,col]),('mean',lambda r:r[0][:,col].mean())]:
  print('RANK',col,k,[(np.round(r[0][:,col],4).tolist(),r[1:]) for r in sorted(R,key=f)[:7]])
print('B QUAD COEFF',ls(quad(h)[:52],Y[:52])[:,[0,-1]].round(4)); print('B OP COEFF DETAILS'); print('AREAS', x.min(0),x.max(0),h.min(0),h.max(0))
