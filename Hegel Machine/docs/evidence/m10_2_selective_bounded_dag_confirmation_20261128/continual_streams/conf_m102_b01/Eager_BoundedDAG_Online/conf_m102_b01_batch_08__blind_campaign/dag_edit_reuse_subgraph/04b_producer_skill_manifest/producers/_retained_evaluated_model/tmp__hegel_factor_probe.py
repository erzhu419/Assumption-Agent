exec(open('/tmp/hegel_panel_structure_probe.py').read().split('for name,x,v in')[0])
from scipy.optimize import least_squares
node=runpy.run_path('/holdout_task/node_01.py'); u,W,C1,C2=node['_cal'](s,w); y=s['train']['y']; T=s['train']['traj']; n=len(y); rng=np.random.default_rng(17); ii=np.arange(n); spl=[(ii[:52],ii[52:])]+[(np.setdiff1d(ii,j),j) for j in np.array_split(rng.permutation(ii),4)]; sv=np.linalg.svd(T,full_matrices=False); v=sv[2][0]; ts=T@v; print('rank1 noise',np.mean(abs(T-ts[:,None]*v)))
A=s['panel_a']; co=a['_fit_panel_a'](A['u'],A['v']); ui=a['_inverse'](A['v'],co); print('A inverse quantiles',np.quantile(abs(ui-A['u']),[.25,.5,.75,.9,.95,.99,1])); print('A keys',a.keys());
for typ in ['rational','tanh','cube','sinh','sin','inv3','exact']:
 def fn(z,g):
  if typ=='rational':return z**3/(1+g*z*z)
  if typ=='tanh':return np.tanh(z*g)
  if typ=='cube':return z**3+g*z
  if typ=='sinh':return np.sinh(z*g)
  if typ=='sin':return np.sin(z*g)
  if typ=='inv3':return np.sinh(np.arcsinh(z*g)/3)
  return np.cbrt(z+np.sqrt(z*z+g))-np.cbrt(-z+np.sqrt(z*z+g))
 for ext in [False,True]:
  def mk(z,g):return np.column_stack((np.ones(len(z)),fn(z,g),z if ext else np.empty((len(z),0))))
  def f(g):
   d=mk(b['u'],g);return (d@ls(d,b['v'])-b['v']).ravel()
  opt=least_squares(f,np.ones(4),bounds=(.001,5.)); print('B',typ,ext,np.sqrt(np.mean(opt.fun**2)),opt.x)
Zs={'u':u,'W':W,'wa':np.arctan(1.25*W),'C1':C1,'C2':C2,'uw':np.c_[u,W],'wc':np.c_[W,C2]}
for label,Z in Zs.items():
 Z=Z/(Z[:n].std(0)+1e-8); p=Z.shape[1]; best=[]
 for typ in ['sin','tanh','cubic','atan','rat','square']:
  fn={'sin':np.sin,'tanh':np.tanh,'cubic':lambda x:x*x*x,'atan':np.arctan,'rat':lambda x:x/(1+x*x),'square':lambda x:x*x}[typ]
  for deg in [1,2]:
   F=pol(Z,deg); Qs=[];errs=[]
   for i,j in spl:
    zz=Z[i]; ff=F[i]; targets=np.c_[y,ts]; yy=targets[i]; G=ff.T@ff+np.eye(ff.shape[1])*0.08
    def design(k,ids):return np.c_[F[ids],fn(k[0]+Z[ids]@k[1:])]
    def fit(k):
     d=design(k,i); return np.linalg.solve(d.T@d+np.diag(np.r_[np.full(ff.shape[1],.08),.015]),d.T@yy)
    def fun(k):return np.r_[((design(k,i)@fit(k)-yy)/yy.std(0)).ravel(),.04*k]
    sol=min((least_squares(fun,q,max_nfev=150,bounds=(-4.,4.)) for q in [np.r_[0,rng.normal(0,.5,p)] for _ in range(5)]),key=lambda x:x.cost); pred=design(sol.x,j)@fit(sol.x); errs.append(np.mean(abs(pred-targets[j]),0)); Qs.append(sol.x)
   e=np.array(errs);best.append((e[0],e.mean(0),typ,deg,Qs[0]))
 print('index',label,*sorted(best,key=lambda q:sum(q[1]*[1,.45]))[:4],sep='\n',flush=True)
