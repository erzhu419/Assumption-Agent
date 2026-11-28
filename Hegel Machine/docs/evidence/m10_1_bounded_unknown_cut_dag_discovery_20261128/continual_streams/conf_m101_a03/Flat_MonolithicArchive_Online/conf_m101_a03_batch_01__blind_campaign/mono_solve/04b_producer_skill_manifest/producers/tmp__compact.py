import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from batch_solver import _affine as af,_fit as fit,_polynomial as poly,_coordinates,solve_batch
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; y=t['y']; w=_coordinates(s,t['x']); x=t['x']; n=len(y)
rng=np.random.default_rng(127); splits=[(np.arange(k),np.arange(k,n)) for k in (52,56)]; splits += [(q[:52],q[52:]) for q in [rng.permutation(n) for _ in range(12)]]; splits += [(np.argsort(w[:,k]*d)[:52],np.argsort(w[:,k]*d)[52:]) for k in range(4) for d in (-1,1)]
def alt(i,j):
 a=af(w); f=poly(w,2); v=poly(w,3); h=fit(a[i],y[i]); d=a@h; rows={}
 def put(k,p): rows[k]=np.mean(abs(p[j]-y[j]))
 b=dict(s,train={key:val[i] if key!='summary' else val for key,val in t.items()},test={'x':x[j]}); pred=solve_batch(b); rows['fixed3']=np.mean(abs(pred['y_hat']-y[j])); rows['traj_worst_time']=np.max(np.mean(abs(pred['state_pred']-t['traj'][j]),axis=0))
 for typ,fn,p0 in [('tanh',np.tanh,1.),('sqrt',lambda q:q/np.sqrt(1+q*q),.8),('softsign',lambda q:q/(1+abs(q)),.3),('atan',np.arctan,1.)]:
  def fitprof(p,g):
   z=np.c_[g,fn(a@p)]; return z@fit(z[i],y[i]),fit(z[i],y[i]);
  for g,label in [(f,'full'),(np.c_[a,w[:,0]*w[:,2],w[:,1]*w[:,3]],'pair')]:
   r=least_squares(lambda p:fitprof(p,g)[0][i]-y[i],h*p0,max_nfev=125); p,c=fitprof(r.x,g); put(typ+'_'+label,p)
   if i[0]==0 and len(i)==52 and label=='full':print(typ,'train rmse',np.std(p[i]-y[i]),'quadratic w',c[5:-1].round(3),'index',r.x.round(3),'amp',round(c[-1],3))
 for k in (.1,.2,.35,1.):
  for dr in (0,.003,.015,.05):
   z=v/(1+k*abs(d[:,None])); L=np.diag(np.r_[np.zeros(5),np.repeat(.15,10),np.ones(20)])*np.sqrt(dr); c=fit(np.r_[z[i],L],np.r_[y[i],np.zeros(35)]);put(f'fixed{k}_{dr}',z@c)
 return rows
out=[alt(i,j) for i,j in splits]; print('model required52 required56 random_mean random_worst directional_worst');
for k in sorted(out[0],key=lambda k:max(r[k] for r in out[2:14])):
 e=np.array([r[k] for r in out]);print(k,np.round([*e[:2],e[2:14].mean(),max(e[2:14]),max(e[14:])],5))
