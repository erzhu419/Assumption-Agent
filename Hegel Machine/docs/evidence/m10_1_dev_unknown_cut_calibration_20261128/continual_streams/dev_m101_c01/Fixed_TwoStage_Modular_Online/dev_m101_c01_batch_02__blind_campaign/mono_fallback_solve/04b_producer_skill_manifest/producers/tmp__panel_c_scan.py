import pickle,numpy as np
from scipy.optimize import least_squares,minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_c']['u']; np.set_printoptions(precision=4,suppress=True)
folds=[(np.arange(len(u))%4!=i,np.arange(len(u))%4==i) for i in range(4)]
def design(q,p,kind):
 f=p[0]
 if kind=='harm': g=np.c_[q,np.sin(f*q),np.cos(f*q)]
 if kind=='sin': g=np.c_[q,np.sin(f*q)]
 if kind=='cos': g=np.c_[q,np.cos(f*q)]
 if kind=='sqharm': g=np.c_[q,q*q,np.sin(f*q),np.cos(f*q)]
 if kind=='sqsin': g=np.c_[q,q*q,np.sin(f*q)]
 if kind=='shift': g=np.c_[q,np.sin(f*q+p[1])]
 if kind=='tanh': g=np.c_[q,np.tanh(f*q)]
 if kind=='squash': g=np.tanh(f*q+p[1])+p[2]*q
 if kind=='lens': g=q+p[1]*np.sin(f*q)+p[2]*(1-np.cos(f*q))
 if kind=='bend': g=q+p[1]*q*q+p[2]*np.sin(f*q)
 if kind=='power': g=np.c_[q,q*q,q**3,q**4]
 if kind=='square': g=np.c_[q,q*q]
 return np.c_[np.ones(len(q)),g]
for k in ('v1','v2'):
 v=s['panel_c'][k]; print(k)
 ans=[]
 for kind in ('harm','sin','cos','sqharm','sqsin','shift','tanh','squash','lens','bend','power','square'):
  def fit(i):
   def res(p):
    X=design(u[i],p,kind);return (X@np.linalg.lstsq(X,v[i],rcond=None)[0]-v[i]).ravel()
   bounds=([.15,-5,-5],[7,5,5]); fits=[least_squares(res,[f,.1,.1],bounds=bounds,max_nfev=100) for f in (1,2.3,3.7,5)]
   p=min(fits,key=lambda r:r.cost).x; c=np.linalg.lstsq(design(u[i],p,kind),v[i],rcond=None)[0];return p,c
  val=[]
  for i,j in folds:
   p,c=fit(i);val.append(np.sqrt(np.mean((design(u[j],p,kind)@c-v[j])**2)))
  p,c=fit(np.ones(len(u),bool)); rmse=np.sqrt(np.mean((design(u,p,kind)@c-v)**2)); ans.append((np.mean(val),kind,p,c));print(kind,round(rmse,6),np.round(val,5),p)
 for _,kind,p,c in sorted(ans,key=lambda a:a[0])[:3]: print(kind,c)
