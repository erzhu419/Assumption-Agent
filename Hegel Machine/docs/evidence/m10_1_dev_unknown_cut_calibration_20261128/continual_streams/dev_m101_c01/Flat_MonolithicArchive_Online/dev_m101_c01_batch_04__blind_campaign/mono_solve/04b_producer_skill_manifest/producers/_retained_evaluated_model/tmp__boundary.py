exec(open('/tmp/structure.py').read().split('for k in')[0])
from sklearn.model_selection import KFold
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
import sys
sys.path.insert(0,'/holdout_task')
from batch_solver import _state,_readout
r=np.argsort(np.linalg.norm(x,axis=1)); folds=[(np.arange(56),np.arange(56,72)),*KFold(5,shuffle=True,random_state=12).split(x),(r[:-12],r[-12:]),*KFold(4,shuffle=True,random_state=327).split(x)]
P=poly(x,2); C=poly(x,3)[:,15:]; D=(x[:,None,:]-x[None,:,:])**2; A=1+x@x.T
for isy in (True,False):
 out={}
 for i,j in folds:
  ztrain=z[i]; mode=np.linalg.svd(ztrain,full_matrices=False)[2][0]; Y=y[i] if isy else ztrain@mode
  old=_readout(x[i],x[j],Y) if isy else _state(x[i],x[j],ztrain)
  def add(key,pred): out.setdefault(key,[]).append(np.mean(abs(pred-(y[j] if isy else z[j]))))
  add('current',old)
  def score(key,p): add(key,p if isy else p[:,None]*mode)
  for coords,w in [('x',x),('a',(x-bias)/scale),('b',poly((x-bias)/scale,1)@cb)]:
   q=poly(w,3)[:,15:]; B=P@P.T; G=q@q.T; d=(w[:,None,:]-w[None,:,:])**2
   for typ in ['rbf','dot','matern']:
    for cubic in [0,.03,.3,2]:
     for n in ([.0002,.006] if isy else [.00004,.0003]):
      def k(h,jj):
       ds=d[jj[:,None],i]@np.exp(h[:4]); R=np.sqrt(5*ds)
       g=((1+R+R*R/3)*np.exp(-R)) if typ=='matern' else np.exp(-ds/2)
       if typ=='dot': g*=A[jj[:,None],i]
       return B[jj[:,None],i]+cubic*G[jj[:,None],i]+np.exp(h[4])*g
      def f(h):
       c=cho_factor(k(h,i)+n*np.eye(len(i)),check_finite=False); a=cho_solve(c,Y,check_finite=False)
       return .5*Y@a+np.log(c[0].diagonal()).sum()+.8*np.sum((h[:4]-h[:4].mean())**2)
      h=minimize(f,[-1.5]*4+[0],bounds=[(-6,2)]*4+[(-8,5)],method='L-BFGS-B',options={'maxiter':45}).x
      c=cho_factor(k(h,i)+n*np.eye(len(i)),check_finite=False); score((coords,typ,cubic,n),k(h,j)@cho_solve(c,Y,check_finite=False))
 def metric(e): return np.mean(e)+.4*max(e)
 print('readout' if isy else 'state','baseline',np.round(out['current'],5),flush=True)
 for key,e in sorted(out.items(), key=lambda r:metric(r[1]))[:12]: print(key,round(metric(e),5),np.round(e,5),flush=True)
