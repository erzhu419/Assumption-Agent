exec(open('/tmp/latent.py').read().split("print('cal'")[0])
from scipy.optimize import minimize_scalar
from scipy.linalg import qr
from sklearn.linear_model import Ridge

def panel(p,key,kind):
 u=p['u']; v=p[key]; funcs={'atan':np.arctan,'tanh':np.tanh,'rat':lambda u:u/np.sqrt(1+u*u),'abs':np.abs,'softabs':lambda u:np.sqrt(1+u*u),'cos':np.cos}
 def design(u,k):
  h=funcs[kind](u*k) if kind not in ('abs',) else abs(u-k)
  return np.c_[aff(u),h]
 def opt(k,ret=False):
  a=design(u,k); Q=np.linalg.qr(a)[0]; e=v-Q@(Q.T@v); err=np.mean((e/(1-(Q*Q).sum(1))[:,None])**2)**.5
  return (err,ls(a,v)) if ret else err
 r=minimize_scalar(opt,bounds=((-0.35,.35) if kind=='abs' else (.35,4)),method='bounded'); return r.fun,r.x,opt(r.x,True)[1],lambda zz:design(zz,r.x)@opt(r.x,True)[1]
ops={}
for pn,key,n in [('panel_b','v','b'),('panel_c','v1','p'),('panel_c','v2','q')]:
 opts=sorted((panel(s[pn],key,k),k) for k in ['atan','tanh','rat','abs','softabs','cos']);
 for (err,k,B,_),kind in opts[:3]: print(n,kind,err,k, '\n',B if kind==opts[0][1] else '')
 ops[n]=opts[0][0][-1]
I=np.arange(52); J=np.arange(52,72); _,_,V=np.linalg.svd(t[I],full_matrices=False); Y=np.c_[y,t@V[0]]
def feat(z):
 b,p,q=(ops[k](z) for k in ['b','p','q']); return dict(z=z,b=b,p=p,q=q,bp=b*p,bq=b*q,pq=p*q,p_q=p-q,pqsum=p+q,bp_sum=b+p,bq_sum=b+q)
F=feat(z)
def evaluate(A):
 m=A[I].mean(0);s=A[I].std(0)+1e-10; A=(A-m)/s; out=[]
 for k in [.000001,.001,.03,.3,2]:
  r=Ridge(alpha=k).fit(A[I],Y[I]); pred=r.predict(A[J]); out.append(np.mean(abs(pred-Y[J]),0)*[1,.94/2])
 return np.array(out)
for n,a in F.items():
 for form in [1,2]: print(n,form,np.round(evaluate(PolynomialFeatures(form,include_bias=False).fit_transform(a)),3).T)
for names in [('z','b'),('z','p'),('z','q'),('p','q'),('b','p','q'),list(F)]:
 A=np.concatenate([F[k] for k in names],axis=1); print(names,'lin',evaluate(A).T)
 if A.shape[1]<=12:print(names,'quad',evaluate(PolynomialFeatures(2,include_bias=False).fit_transform(A)).T)
for j in range(4):
 ui=u[g,j]; ni=nxt=np.roll(u,-1,axis=1)[g,j]; resid=v[g,j]-(np.c_[np.ones(len(ui)),ui,ni,ui*ni]@c[:,j]);print('A resid',j,np.std(resid),np.round(resid[:10],3))
