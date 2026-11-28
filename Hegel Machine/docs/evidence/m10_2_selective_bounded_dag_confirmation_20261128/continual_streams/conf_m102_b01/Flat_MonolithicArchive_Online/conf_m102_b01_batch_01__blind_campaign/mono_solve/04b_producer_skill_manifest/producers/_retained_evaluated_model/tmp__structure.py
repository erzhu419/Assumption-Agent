import pickle,numpy as np,itertools
from scipy.optimize import least_squares,minimize_scalar
from scipy.special import expit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];np.set_printoptions(precision=5,suppress=True,linewidth=140)
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def a_cal(x):
 a=s['panel_a'];out=[]
 for u,v,z in zip(a['u'].T,a['v'].T,x.T):
  D=aff(u); b=ls(D,v)
  for _ in range(8):
   j=np.argsort(abs(D@b-v))[:41];b=ls(D[j],v[j])
  j=abs(D@b-v)<.002;b=ls(D[j],v[j]);out.append((z-b[0])/b[1])
 return np.array(out).T
def poly(x,d):return np.column_stack([np.prod(x[:,j],axis=1) for i in range(d+1) for j in itertools.combinations_with_replacement(range(4),i)])
fs={'tanh':np.tanh,'sin':np.sin,'asinh':np.arcsinh,'atan':np.arctan,'inv3':lambda x:np.sinh(np.arcsinh(x)/3),'cubic':lambda x:x+x**3,'rat':lambda x:x/(1+x*x),'softsign':lambda x:x/np.sqrt(1+x*x),'exp':np.exp,'logcosh':lambda x:np.logaddexp(x,-x)-np.log(2)}
if __name__=='__main__':
 for k,j in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  x=s[k]['u']; y=s[k][j];D=aff(x);print('\nPANEL',k,j)
  for d in [1,2,3]:
   F=poly(x,d);R=y-F@ls(F,y); print('poly',d,np.std(R,0), 'loo',np.std(R/np.maximum(1e-8,1-np.sum(F*np.linalg.pinv(F).T,axis=1))[:,None],0))
  print('res sv',np.linalg.svd(y-D@ls(D,y),compute_uv=False));scores=[]
  for name,f in fs.items():
   for add in [False,True]:
    grid=np.geomspace(.05,8,50);best=[]
    for g in grid:
     F=aff(f(g*x)) if not add else np.c_[D,f(g*x)]; R=y-F@ls(F,y);best.append(np.mean(R*R)**.5)
    scores.append((min(best),name,'input+lin' if add else 'input',grid[np.argmin(best)]))
   errs=[];pars=[]
   for l,z in enumerate(y.T):
    b=ls(D,z);opts=[]
    for g in [.3,1,2,4]:
     for add in [0,1]:
      def pred(p):return p[0]+p[1]*f(D@p[2:7])+add*(D@p[7:])
      r=least_squares(lambda p:pred(p)-z,np.r_[0,1/g,b*g,np.zeros(5)],max_nfev=200);opts.append((np.std(r.fun),add,r.x))
    e,ad,p=min(opts,key=lambda z:z[0]);errs.append(e);pars.append((ad,np.round(p,3)))
   scores.append((np.linalg.norm(errs)/2,name,'output',np.round(errs,5)))
  for v in sorted(scores,key=lambda z:z[0])[:14]:print(v)
 tr=s['train']; X=a_cal(tr['x']);Y=tr['y'];T=tr['traj'];t=s['eval_times']; _,_,V=np.linalg.svd(T,full_matrices=False);print('uncentered sv',np.linalg.svd(T,compute_uv=False),'modes',V[:2]);
 for fn in [lambda a,k:1-np.exp(-a*k),lambda a,k:np.log1p(a*k),lambda a,k:k/(1+a*k)]:
  def tp(v):return np.array([fn(v[0],t),t]).T
  def e(v):D=tp(v);return (T.T-D@ls(D,T.T)).ravel()
  opt=least_squares(e,[1.],bounds=(.00001,100));D=tp(opt.x); Q=ls(D,T.T).T;print('time basis',opt.x,np.std(opt.fun), 'Q bounds',Q.min(0),Q.max(0));print('Q head',Q[:12]);
  for d in [1,2,3]:
   F=poly(X,d);w=F@ls(F[:52],np.c_[Y,Q][:52]);print('polyreg',d,np.std(w[:52]-np.c_[Y,Q][:52],0),np.mean(abs(w[52:]-np.c_[Y,Q][52:]),0))
