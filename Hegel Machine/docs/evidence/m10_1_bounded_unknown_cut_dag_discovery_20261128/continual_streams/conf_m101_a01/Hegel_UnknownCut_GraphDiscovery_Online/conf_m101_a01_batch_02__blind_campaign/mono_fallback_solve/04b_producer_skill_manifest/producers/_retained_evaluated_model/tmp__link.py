from explore import *
from scipy.optimize import least_squares
from scipy.linalg import lstsq
folds=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72))]+[(np.setdiff1d(np.arange(72),j),j) for j in np.array_split(np.random.default_rng(26).permutation(72),4)]
forms={'tanh':np.tanh,'asinh':np.arcsinh,'softsign':lambda a:a/(1+abs(a)),'sqrt':lambda a:a/np.sqrt(1+a*a),'atan':np.arctan,'logistic':lambda a:np.logaddexp(0,a),'rational':lambda a:a/(1+a*a),'cubic':lambda a:a-a**3,'exp':lambda a:np.expm1(np.clip(a,-10,8)),'clip':lambda a:np.clip(a,-1,1)}
Q=PolynomialFeatures(2); P=Q.fit_transform(x)
def fit(F,yy,G,name,reg=0.):
 g=forms[name]; b=lstsq(F,yy)[0]; eBest=np.inf
 for scale in [.3,.8,1.8]:
  for mode in [0,1]:
   def predict(h,r):return h[-2]+np.exp(h[-1])*g(r@h[:-2])+(r[:,1:5]@h[-6:-2] if False else 0)
   def err(h):return np.r_[predict(h,F)-yy, np.sqrt(reg)*h[1:-2]]
   h=np.r_[b/scale*(1 if mode==0 else .45),0,np.log(scale)]; h=least_squares(err,h,max_nfev=160,bounds=(np.r_[np.full(len(b),-np.inf),-4,-4],np.r_[np.full(len(b),np.inf),4,4]),ftol=1e-7).x
   e=np.linalg.norm(err(h));
   if e<eBest:eBest=e; result=(predict(h,G),h,e/np.sqrt(len(yy)))
 return result
for n in forms:
 result=[fit(P[tr],y[tr],P[te],n,1e-6) for tr,te in folds]; a=[np.mean(abs(pr-y[te])) for (pr,_,_),(tr,te) in zip(result,folds)]; print(n,np.round(a,5),'train',round(result[0][2],5),'offset/amp',np.round([result[0][1][-2],np.exp(result[0][1][-1])],4),flush=True)
for d in [3,4,5]:
 H=PolynomialFeatures(d).fit_transform(x); powers=PolynomialFeatures(d).fit(x).powers_; deg=powers.sum(1); results=[]
 for a in [.001,.01,.05,.2,1,5,20,100]:
  for decay in [1,4,16]:
   for ix in [False,True]:
    costs=(a*decay**(np.maximum(deg,3)-3))*np.where(deg<3,.001,1.)*np.where(ix,np.maximum(1,(powers>0).sum(1))**2,1.); es=[]
    for tr,te in folds:
     b=np.linalg.solve(H[tr].T@H[tr]+np.diag(costs),H[tr].T@y[tr]); es.append(np.abs(H[te]@b-y[te]).mean())
    results.append((max(es),np.mean(es),es[0],a,decay,ix))
 print('poly',d,sorted(results)[:4],flush=True)
