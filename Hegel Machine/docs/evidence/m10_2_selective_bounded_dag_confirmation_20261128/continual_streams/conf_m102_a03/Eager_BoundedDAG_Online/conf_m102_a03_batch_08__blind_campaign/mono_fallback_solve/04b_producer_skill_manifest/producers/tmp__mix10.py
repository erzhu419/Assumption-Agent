exec(open('/tmp/graph7.py').read().split("B=auxiliary")[0]); from scipy.optimize import least_squares
from itertools import combinations
pair=lambda u:np.array([u[:,i]*u[:,j] for i,j in combinations(range(4),2)]).T
D={}
for pn,j in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 U,V=s[pn]['u'],s[pn][j]; n=U.shape[1]; S=[]
 fs={'tanh':np.tanh,'atan':np.arctan,'sin':np.sin,'cos':np.cos,'rat':lambda x:x/(1+x*x),'sqrt':lambda x:x/np.sqrt(1+x*x),'cubic':lambda x:x**3,'rat3':lambda x:x**3/(1+x*x),'softplus':lambda x:np.logaddexp(x,0),'sinh':np.sinh}
 for mode,f in fs.items():
  for extras in ('none','lin','linpair'):
   def bas(u):return {'none':np.ones((len(u),1)),'lin':aff(u),'linpair':aff(np.c_[u,pair(u)])}[extras]
   A=bas(U); results=[]; out=[]
   for k in range(4):
    v=V[:,k]; d=ls(aff(U),v); w=d[1:]/np.linalg.norm(d[1:]);
    def calc(q,u=U):return np.c_[bas(u),f(np.clip(u@q[:4]+q[4],-8,8))]
    def err(q):
     X=calc(q);return X@ls(X,v)-v
    ini=[np.r_[sc*w,b] for sc in [.5,1.5,3] for b in [0,.5,-.5]]
    r=min((least_squares(err,q,max_nfev=100,bounds=(-7,7)) for q in ini),key=lambda h:np.linalg.norm(h.fun));results.append(r.fun);out.append(calc(r.x,z)@ls(calc(r.x),v))
   err0=np.std(results);S.append((err0,mode,extras)); D[pn,j,mode,extras]=np.array(out).T
 print(pn,j,'postmix',sorted(S)[:7]);
 for e,mode,ex in sorted(S)[:3]:
  h=D[pn,j,mode,ex]; F=PolynomialFeatures(2).fit_transform(h); R=abs(F[52:]@ls(F[:52],Y[:52])-Y[52:]);print('down',mode,ex,R[:,0].mean(),R[:,1:].mean())
# Try componentwise shifted functions before mixing with optional genuine interactions.
U,V=s['panel_b']['u'],s['panel_b']['v']; M=[]
for name,f in fs.items():
 for ex in ('none','lin','pair','sq','linpair','full'):
  def feat(u,p):
   b={'none':np.empty((len(u),0)),'lin':u,'pair':pair(u),'sq':u*u,'linpair':np.c_[u,pair(u)],'full':np.c_[u,u*u,pair(u)]}[ex];return aff(np.c_[b,f(np.clip(u*np.exp(p[:4])+p[4:],-8,8))])
  def err(p):
   X=feat(U,p);return (X@ls(X,V)-V).ravel()
  r=min((least_squares(err,np.r_[np.full(4,a),np.full(4,b)],bounds=(-4,2),max_nfev=120) for a,b in [(0,0),(.7,0),(-.6,.15)]),key=lambda r:np.linalg.norm(r.fun)); A=feat(U,r.x); h=feat(z,r.x)@ls(A,V); F=PolynomialFeatures(2).fit_transform(h); E=abs(F[52:]@ls(F[:52],Y[:52])-Y[52:]); M.append((np.std(r.fun),E[:,0].mean(),E[:,1:].mean(),name,ex,np.round(r.x,3)))
print('premix panel',sorted(M,key=lambda a:a[0])[:10]);print('premix y',sorted(M,key=lambda a:a[1])[:10])
