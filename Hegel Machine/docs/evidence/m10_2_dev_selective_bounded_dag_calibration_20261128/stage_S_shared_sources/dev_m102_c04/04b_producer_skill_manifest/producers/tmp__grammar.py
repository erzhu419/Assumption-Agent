import numpy as np,pickle
from scipy.optimize import least_squares
from scipy.linalg import qr
from itertools import product
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,z=[s['train'][k] for k in ('x','y','traj')]; t=s['eval_times']; n=len(x)
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(f,v,lam=0):
 if not lam:return np.linalg.lstsq(f,v,rcond=1e-10)[0]
 return np.linalg.solve(f.T@f+lam*np.eye(f.shape[1]),f.T@v)
a=s['panel_a']; mask=np.linalg.norm(aff(a['u'])@ls(aff(a['u']),a['v'])-a['v'],axis=1)<2; da=ls(aff(a['u'][mask]),a['v'][mask]); xa=(x-da[0])/np.diag(da[1:]); xf=aff(x)@da
b=s['panel_b']; db=ls(aff(b['u']),b['v']); c=s['panel_c']; u=c['u'];
fun={'sin':lambda a,k:np.sin(k*a),'tanh':lambda a,k:np.tanh(k*a),'cos':lambda a,k:np.cos(k*a)-1,'rat':lambda a,k:a/(1+k*a*a),'rat2':lambda a,k:a*a/(1+k*a*a),'cubic':lambda a,k:a**3,'square':lambda a,k:a*a,'abs':lambda a,k:abs(a),'exp':lambda a,k:np.exp(-k*a*a)-1,'soft':lambda a,k:a/np.sqrt(1+k*a*a)}
models=[]
for j in ('v1','v2'):
 v=c[j]; print(j,'options:'); scores=[]
 for key,g in fun.items():
  for typ in ('add','only','post','joint'):
   if typ in ('post','joint') and key not in ('sin','tanh','rat','soft'): continue
   if typ in ('add','only'):
    def F(x,k):return np.c_[aff(x),g(x,k)] if typ=='add' else aff(g(x,k))
    opt=least_squares(lambda k:(F(u,k[0])@ls(F(u,k[0]),v)-v).ravel(),[.8],bounds=(.01,4),max_nfev=60).x[0]; w=ls(F(u,opt),v); err=np.mean(abs(F(u,opt)@w-v)); predict=lambda q,op=opt,w=w,g=g,typ=typ:(np.c_[aff(q),g(q,op)] if typ=='add' else aff(g(q,op)))@w
   else:
    w=ls(aff(u),v)
    def P(q,p):
     m=aff(q)@p[:20].reshape(5,4); k=p[20]; out=g(m,k)
     return out if typ=='post' else m+p[21:]*out
    ini=np.r_[w.ravel(),.8] if typ=='post' else np.r_[w.ravel(),.8,np.full(4,.15)]; low=np.r_[np.full(20,-np.inf),.01,np.full(len(ini)-21,-np.inf)]; high=np.r_[np.full(20,np.inf),4,np.full(len(ini)-21,np.inf)]
    p=least_squares(lambda p:(P(u,p)-v).ravel(),ini,bounds=(low,high),max_nfev=150).x; opt=p[20]; err=np.mean(abs(P(u,p)-v)); predict=lambda q,p=p,g=g,typ=typ:g(aff(q)@p[:20].reshape(5,4),p[20]) if typ=='post' else aff(q)@p[:20].reshape(5,4)+p[21:]*g(aff(q)@p[:20].reshape(5,4),p[20])
   scores.append((err,key,typ,opt,predict))
 scores.sort(key=lambda a:a[0]); print([v[:4] for v in scores[:10]]); models.append(scores[:3]);
# empirical temporal basis and candidate analytic shapes
for name,TF in [('exp+exp',lambda p:np.stack([-np.expm1(-t*p[0]),-np.expm1(-t*p[1])],1)),('exp+decay',lambda p:np.stack([-np.expm1(-t*p[0]),t*np.exp(-p[1]*t)],1)),('exp+sin',lambda p:np.stack([-np.expm1(-t*p[0]),np.sin(p[1]*t)],1)),('exp+linear',lambda p:np.stack([-np.expm1(-t*p[0]),t],1)),('exp+t-exp',lambda p:np.stack([-np.expm1(-t*p[0]),t+np.expm1(-t*p[0])/p[0]],1)),('ode',lambda p:np.stack([-np.expm1(-t*p[0]),-np.expm1(-t*p[0])+(np.exp(-t*p[1])-np.exp(-t*p[0]))*p[0]/(p[0]-p[1])],1))]:
 f=lambda p:(TF(p)@ls(TF(p),z[:52].T)-z[:52].T).ravel(); p=least_squares(f,[1.25,.55],bounds=(.01,6)).x; F=TF(p); print('time',name,p,np.mean(abs(f(p))),np.mean(abs(F@ls(F,z[52:].T)-z[52:].T)))
_,_,vv=np.linalg.svd(z[:52],full_matrices=False); target=np.c_[y,z@vv[:2].T]
def report(F,name,output):
 for lam in (1e-8,1e-4,1e-2):
  w=ls(F[:52],target[:52],lam); e=F[52:]@w-target[52:]; output.append((np.mean(abs(e[:,0])),np.mean(abs(e[:,1:]@vv[:2])),name,lam))
out=[]
for label,q in [('x',x),('Ainv',xa),('Afwd',xf),('B(x)',aff(x)@db),('B(Ainv)',aff(xa)@db)]:
 for i,j in product(range(3),repeat=2):
  p,r=models[0][i][-1](q),models[1][j][-1](q); dr=f'{label} C{i}{j}'
  for tag,F in [('base',np.c_[aff(q),p,r]),('alt',np.c_[aff(q),p,r,np.tanh(p),np.tanh(r),np.sin(q)]),('pair',np.c_[aff(q),p,r,np.tanh(p),p*r,p*p,r*r]),('solo',np.c_[aff(p),r])]:report(F,dr+' '+tag,out)
 for key,g in fun.items():
  for k in (.5,1,1.5,2,3):
   h=g(q,k); report(np.c_[aff(q),h,q*q,q*h],f'{label} {key} {k}',out)
print('best y',sorted(out)[:14]);print('best traj',sorted(out,key=lambda e:e[1])[:14]);
