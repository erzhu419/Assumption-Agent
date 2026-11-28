exec(open('/tmp/probe6.py').read().split('for xx,n')[0]); from scipy.optimize import minimize
from itertools import combinations
from sklearn.linear_model import Lasso
for name,fn,guess in [('tanh',np.tanh,.85),('atan',np.arctan,1.),('soft',lambda x:x/(1+abs(x)),.6),('soft2',lambda x:x/np.sqrt(1+x*x),.8),('sin',np.sin,.8),('gauss',lambda x:x*np.exp(-x*x),.25)]:
 for dat,pow in [(v,1),(w,2)]:
  def f(p,xx):return h(fn(xx*np.exp(p[:4])+p[4:])**pow)
  j=np.arange(len(u))%5!=0
  res=least_squares(lambda p:(f(p,u[j])@fit(f(p,u[j]),dat[j])-dat[j]).ravel(),np.r_[np.full(4,np.log(guess)),np.zeros(4)],bounds=(-4,3)); print(name,pow,np.round(np.r_[np.exp(res.x[:4]),res.x[4:]],4),rms(res.fun),rms(f(res.x,u[~j])@fit(f(res.x,u[j]),dat[j])-dat[~j]))

def bases(xx):
 yield 'rawquad',poly(xx,2)
 for scale in [.4,.6,.8,1.,1.2,1.5,2.]:
  for name,fn in [('tanh',np.tanh),('atan',np.arctan),('sin',np.sin),('soft',lambda a:a/(1+abs(a))),('soft2',lambda a:a/np.sqrt(1+a*a))]:
   tt=fn(scale*xx); yield (name,scale,2),poly(tt,2); yield (name,scale,3),poly(tt,3); yield (name,scale,'mix'),np.c_[poly(xx,2),tt,tt**2,tt**3,*[tt[:,i]*tt[:,j] for i,j in combinations(range(4),2)]]
sp=[(np.arange(52),np.arange(52,72))]+[(np.where(np.arange(72)%4!=j)[0],np.where(np.arange(72)%4==j)[0]) for j in range(4)]
for xn,xx in [('z',z),('Bz',z@B[1:]+B[0]),('unBz',(z-B[0])@np.linalg.inv(B[1:]))]:
 scores=[]
 for nm,F in bases(xx):
  for reg in [0,.00001,.001,.01,.1]:
   ee=[]
   for i,j in sp:
    mu=F[i,1:].mean(0); sig=F[i,1:].std(0).clip(1e-9); f=(F[:,1:]-mu)/sig; P=f[j]@np.linalg.solve(f[i].T@f[i]+max(reg,1e-10)*np.eye(f.shape[1]),f[i].T@(Y[i]-Y[i].mean(0)))+Y[i].mean(0); ee.append(np.abs(P-Y[j]).mean(0))
   scores.append((np.array(ee),nm,reg))
 for k in range(3):
  print(xn,k,[(np.round(e[:,k],4),nm,reg) for e,nm,reg in sorted(scores,key=lambda a:a[0][:,k].max())[:5]])
