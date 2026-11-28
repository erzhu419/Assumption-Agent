import pickle,numpy as np,itertools
from scipy.optimize import minimize_scalar,least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(s['panel_b'][k] for k in ('u','v'))
F={'sin':lambda x,a:np.sin(a*x),'tanh':lambda x,a:np.tanh(a*x),'rat':lambda x,a:x/(1+a*x*x),'xexp':lambda x,a:x*np.exp(-a*x*x),'atan':lambda x,a:np.arctan(a*x),'softsign':lambda x,a:x/(1+a*abs(x)),'erf':lambda x,a:erf(a*x)}
from scipy.special import erf

def search(mask, verbose=False):
 x,y=u[mask],v[mask]; A=np.c_[np.ones(len(x)),x]; Q=np.linalg.qr(A)[0]; P=np.eye(len(x))-Q@Q.T; R=P@y; T=np.linalg.qr(R)[0]; N=P-T@T.T
 candidates=[]
 for j in range(x.shape[1]):
  def profile(f,a):
   z=f(x[:,j],a); return np.linalg.norm(N@z)**2/max(np.linalg.norm(P@z)**2,1e-24)
  row=[]
  for name,f in F.items():
   g=np.geomspace(.12,7,55); k=np.argmin([profile(f,a) for a in g]); m=minimize_scalar(lambda a:profile(f,a),bounds=(g[max(0,k-1)],g[min(len(g)-1,k+1)]),method='bounded'); row.append((m.fun,name,m.x))
  row.sort(); candidates.append(row)
  if verbose: print('axis',j,[(round(e,7),k,round(a,5)) for e,k,a in row])
 def fit(names,scale):
  B=np.column_stack([F[k](x[:,j],a) for j,(k,a) in enumerate(zip(names,scale))]); Z=np.c_[A,B]; C=np.linalg.lstsq(Z,y,rcond=None)[0]; return Z@C-y,C
 trials=[]
 for spec in itertools.product(*candidates):
  names=tuple(t[1] for t in spec); a=np.array([t[2] for t in spec]); trials.append((np.linalg.norm(fit(names,a)[0]),names,a))
 trials.sort(key=lambda t:t[0]); final=[]
 for _,names,a in trials[:20]:
  opt=least_squares(lambda log_a:fit(names,np.exp(log_a))[0].ravel(),np.log(a),bounds=(np.log(.05),np.log(10)),max_nfev=60,gtol=1e-12,ftol=1e-12,xtol=1e-12); a=np.exp(opt.x); e,C=fit(names,a); final.append((np.linalg.norm(e),names,a,C,e))
 final.sort(key=lambda t:t[0]);
 if verbose:
  for e,ns,a,_,_ in final[:8]: print('joint',round(e,9),ns,np.round(a,6))
  for name in F:
   ns=(name,)*4; a=np.array([next(t[2] for t in row if t[1]==name) for row in candidates]); opt=least_squares(lambda z:fit(ns,np.exp(z))[0].ravel(),np.log(a),bounds=(np.log(.05),np.log(10))); a=np.exp(opt.x); print('homogeneous',name,np.linalg.norm(fit(ns,a)[0]),a)
  print('best coefficients',np.round(final[0][3],6)); print('best residual by row',np.round(np.linalg.norm(final[0][4],axis=1),6))
 return final[0]
if __name__=='__main__':
 search(np.ones(len(u),bool),True)
 err=[]; config={}
 for k in range(len(u)):
  mask=np.arange(len(u))!=k; _,ns,a,C,_=search(mask); config[ns]=config.get(ns,0)+1; z=np.r_[1,u[k],[F[name](u[k,j],w) for j,(name,w) in enumerate(zip(ns,a))]]; err.append(z@C-v[k])
 print('nested LOO family counts',config,'MAE',np.mean(np.abs(err)),'max',np.max(np.abs(err)))
