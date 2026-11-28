import pickle,numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']; M=np.c_[y,z]; n=52
A=[]
for k in ['panel_a','panel_b','panel_c']:
 p=s[k]; u=p['u']; v=p.get('v',p.get('v1')); H=np.c_[np.ones(len(u)),u]; w=np.linalg.lstsq(H,v,rcond=None)[0];
 if k=='panel_a':
  for i in range(10):
   j=np.argsort(np.linalg.norm(H@w-v,axis=1))[:39]; w=np.linalg.lstsq(H[j],v[j],rcond=None)[0]
 A.append(w)
xf=(x-A[0][0])/A[0][1:].diagonal(); xb=(np.c_[np.ones(len(x)),xf]@A[1]); xc=(np.c_[np.ones(len(x)),xf]@A[2]);
I=np.arange(len(x)); folds=[I[52:],np.argsort(np.linalg.norm(x,axis=1))[-20:],np.argsort(abs(x).max(1))[-20:],I[:20]]+[np.random.default_rng(i).choice(72,20,False) for i in range(8)]
def scores(F,reg):
 R=[]; K=F@F.T
 for te in folds:
  tr=np.setdiff1d(I,te); rr=M[te]-K[np.ix_(te,tr)]@np.linalg.solve(K[np.ix_(tr,tr)]+reg*np.eye(len(tr)),M[tr]); e=abs(rr); R.append([e[:,0].mean(),e[:,1:].mean(),e[:,1:].mean(0).max()])
 R=np.array(R);return np.array([R[0],R.max(0),R.mean(0)])
f=PolynomialFeatures(5); raw=f.fit_transform(x); d=f.powers_.sum(1)
for c,a,l in [(1,0,.006),(.5,0,.012),(.5,.075,.012),(.25,.15,.004),(.25,.3,.004),(.25,.15,.012)]:
 print('cubic grid',c,a,l,scores(raw*(c**np.minimum(np.maximum(d-2,0),1)*a**np.maximum(d-3,0)),l).round(5).tolist())
rows=[]
def test(label,F,regs=(.001,.012,.05)):
 for r in regs:
  T=scores(F,r);rows.append((T[2].sum(),label,r,T))
for name,X in [('raw',x),('cal',xf),('B',xb),('C',xc)]:
 for fac in [.3,.6,1,1.5]:
  for fn,fun in [('sin',np.sin),('tanh',np.tanh),('asinh',np.arcsinh),('rational',lambda a:a/(1+a*a)),('softsign',lambda a:a/np.sqrt(1+a*a))]:
   xx=fun(fac*X)/fac
   for deg in [2,3]:
    F=PolynomialFeatures(deg).fit_transform(xx);test((name,fac,fn,deg),F)
 for fac in [.3,.8,1.3]:
  P=PolynomialFeatures(2).fit_transform(X); dX=np.sin(fac*X)/fac-X
  test((name,fac,'sin_add'),np.c_[P,4*dX]);test((name,fac,'sin_cross'),np.c_[P,4*dX,(X[:,:,None]*dX[:,None,:]).reshape(len(X),-1)])
for _,lab,r,T in sorted(rows,key=lambda a:a[0])[:14]:print('structured',lab,r,T.round(5).tolist())
# Fit a constrained rational correction, using one denominator per scalar and temporal amplitude.
B=np.c_[-np.expm1(-1.533*t),np.log1p(1.01*t)]; U=np.linalg.lstsq(B,z.T,rcond=None)[0].T; Y=np.c_[y,U]; P=raw[:,d<=3]; reg=.005*np.where(d[d<=3]==3,4,1)
for typ in ['quadratic','rankone','radial']:
 out=[]
 for te in folds:
  tr=np.setdiff1d(I,te); pred=[]
  for target in Y.T:
   def denom(p):
    if typ=='quadratic':return np.exp(np.clip(raw[:,d==2]@p,-2,2))
    if typ=='rankone':return 1+(x@p)**2
    return 1+p[0]*np.sum(x*x,1)
   def coeff(p):
    F=P[tr]/denom(p)[tr,None]; return np.linalg.solve(F.T@F+np.diag(reg),F.T@target[tr])
   def res(p):
    w=coeff(p); return np.r_[P[tr]@w/denom(p)[tr]-target[tr],np.sqrt(reg)*w,.06*p]
   nparam={'quadratic':10,'rankone':4,'radial':1}[typ]; init=np.full(nparam,.03); bounds=(0,1) if typ=='radial' else (-.7,.7)
   w=least_squares(res,init,bounds=bounds,max_nfev=55).x; pred.append(P[te]@coeff(w)/denom(w)[te])
  q=np.array(pred).T; ez=abs(q[:,1:]@B.T-z[te]);out.append([abs(q[:,0]-y[te]).mean(),ez.mean(),ez.mean(0).max()])
 R=np.array(out);print('denominator',typ,np.array([R[0],R.max(0),R.mean(0)]).round(5).tolist())
