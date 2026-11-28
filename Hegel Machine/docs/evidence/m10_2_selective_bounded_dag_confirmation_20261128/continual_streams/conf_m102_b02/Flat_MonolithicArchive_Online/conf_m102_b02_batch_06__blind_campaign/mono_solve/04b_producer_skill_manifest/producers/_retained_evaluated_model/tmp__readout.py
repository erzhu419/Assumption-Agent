from graph import *
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import least_squares
np.set_printoptions(suppress=True,precision=4)
w=ops['sat'](z);q=qop(w);r=rop(w);rz=rop(z)
y=y[:72];z,w,q,r,rz=[a[:72] for a in (z,w,q,r,rz)];tr0=tr-tr[:52].mean(0);U,S,V=np.linalg.svd(tr0[:52],full_matrices=False);T=tr0@V.T;print('time',s['eval_times'],'S',S,'V',V[:3]);print('corr T',np.corrcoef(np.c_[y,T].T)[0]);
def score(d):return scores(I(d))
for name,F in [('z',z),('w',w),('zq',np.c_[z,q]),('wq',np.c_[w,q]),('zwq',np.c_[z,w,q]),('zr',np.c_[z,r]),('zqr',np.c_[z,q,r]),('T',T[:,:3]),('zqT',np.c_[z,q,T[:,:3]])]:
 print(name,score(F));
 for k in (2,3,4):
  D=PolynomialFeatures(k,include_bias=False).fit_transform(F); sd=D[:52].std(0); D=D/np.maximum(sd,1e-5); out=[]
  for a in (.0001,.001,.01,.05,.2,1.,4.):
   m=Ridge(alpha=a).fit(D[:52],y[:52]);out.append(np.abs(m.predict(D[52:])-y[52:]).mean())
  print(' polynomial',k,np.round(out,4))
for bn in ('tanh','sat','sin','lin'):
 w=ops[bn](z)
 for label,F in [('z',z),('w',w),('zq',np.c_[z,qop(w)]),('wq',np.c_[w,qop(w)]),('zw',np.c_[z,w]),('zqr',np.c_[z,qop(w),rop(w)]),('wqr',np.c_[w,qop(w),rop(w)]),('zqT',np.c_[z,qop(w),T[:,:3]])]:
  d=I(F); beta=ls(d[:52],y[:52]); ans=[]
  for n,fn in [('tanh',np.tanh),('sin',np.sin),('asinh',np.arcsinh),('sat',lambda a:a/(1+a*a)),('square',lambda a:a*a),('cubic',lambda a:a**3),('abs',np.abs)]:
   def obj(p,dd):return dd@p[:len(beta)]+p[-1]*fn(dd@p[len(beta):-1]);
   fits=[least_squares(lambda p: obj(p,d[:52])-y[:52],np.r_[beta,beta*m,amp],max_nfev=150) for m,amp in ((.65,.5),(1.,-.5),(2.,.5),(.65,-.5))];f=min(fits,key=lambda f:f.cost);ans.append((round(np.mean(abs(obj(f.x,d[52:])-y[52:])),4),round(np.mean(abs(f.fun)),4),n))
  print(bn,label,sorted(ans)[:3])
