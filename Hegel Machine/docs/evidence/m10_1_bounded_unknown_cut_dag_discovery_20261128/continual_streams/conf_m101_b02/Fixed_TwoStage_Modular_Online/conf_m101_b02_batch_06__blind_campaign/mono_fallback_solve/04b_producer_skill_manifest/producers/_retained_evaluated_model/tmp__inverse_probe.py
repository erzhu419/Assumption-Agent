exec(open('/tmp/probe.py').read().split('print(\'A RMSE')[0])
from scipy.optimize import root,least_squares
np.set_printoptions(precision=5,suppress=True,linewidth=160)
xx=np.vstack((x,s['test']['x'])); sol=[root(lambda z:A(z[None])[0]-v,v) for v in xx]; ai=np.array([r.x for r in sol]); print('inverse errors', np.sort(np.linalg.norm(A(ai)-xx,axis=1))[-10:]); print('latent ranges',ai.min(0),ai.max(0));
u=s['panel_b']['u']; v=s['panel_b']['v'];
funcs={'tanh':np.tanh,'sin':np.sin,'erf':__import__('scipy').special.erf,'exp':lambda z:-np.sign(z)*np.expm1(-abs(z)),'rootrat':lambda z:z/np.sqrt(1+z*z),'atan':np.arctan}
bs={}
for name,f in funcs.items():
 for lin in (False,True):
  def g(z,k):return np.c_[des(z) if lin else np.ones(len(z)), f(z*k)]
  def loss(k):
   z=g(u,k);return (z@ls(z,v)-v).ravel()
  r=min((least_squares(loss,np.full(4,k),bounds=(.05,6)) for k in (.5,1.25,2,3,4.5)),key=lambda r:sum(r.fun**2)); print('B basis',name,lin,r.x,np.mean(r.fun**2)**.5)
  b=ls(g(u,r.x),v); bs[name+str(lin)]=(np.mean(r.fun**2)**.5, g(ai,r.x)@b)
# An affine map inside a saturating output, possibly with a linear skip.
for name,f in funcs.items():
 for skip in (False,True):
  X=des(u); Z=des(ai); preds=[]; errs=[]
  for y in v.T:
   w=ls(X,y)
   def pred(X,p):return p[0]+p[1]*f(X@p[2:7])+(X[:,1:]@p[7:] if skip else 0)
   r=min((least_squares(lambda p:pred(X,p)-y,np.r_[0.,1.,w*k,np.zeros(4) if skip else []],max_nfev=400) for k in (.5,1.5,3)),key=lambda r:sum(r.fun**2));preds.append(pred(Z,r.x)); errs.extend(r.fun)
  e=np.mean(np.square(errs))**.5;print('B mixed',name,skip,e);bs['mix'+name+str(skip)]=(e,np.array(preds).T)
d=s['panel_c']; k=minimize_scalar(lambda k:np.linalg.norm((lambda z:z@ls(z,d['v2'])-d['v2'])(des(d['u'],d['u']**2/np.sqrt(1+k*d['u']**2)))),bounds=(.01,2),method='bounded').x;D=fit(lambda u:des(u,u**2/np.sqrt(1+k*u*u)),'panel_c','v2')
def err(z,Y=Y):
 mt=z[:52].mean(0);sd=np.maximum(.00001,z[:52].std(0)); z=(z-mt)/sd; a,w,b=np.linalg.svd(z[:52],full_matrices=False); E=[]
 for reg in (1e-7,.001,.03,.3):
  p=Y[:52].mean(0)+((z[52:72]@b.T)*(w/(w*w+reg)))@(a.T@(Y[:52]-Y[:52].mean(0))); e=np.abs(p-Y[52:]);E.append(np.round([e[:,0].mean(),e[:,1:].mean()],4))
 return np.array(E).tolist()
for key,z in [('inverse',ai)]+[(n,z) for n,(e,z) in sorted(bs.items(),key=lambda p:p[1][0])[:6]]:
 if np.max(np.abs(z))>20:continue
 c=C(z); d=D(z); cd=lambda a,b:(a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
 for n,F in [('lin', z),('poly',np.c_[z,q(z),z**3]),('panels',np.c_[z,c,d]),('cross',np.c_[z,c,d,cd(z,c),cd(z,d)]),('branch',np.c_[z,c,d,cd(c,d)]),('pre/post',np.c_[ai,z,C(ai),D(ai),c,d])]:print(key,n,err(F))
t=s['train']['traj']; times=s['eval_times']; rates=[]; amps=[]
for row in t:
 def prof(k):
  z=np.c_[np.ones(len(times)),-np.expm1(-times*k)];b=ls(z,row);return np.linalg.norm(z@b-row),b
 k=minimize_scalar(lambda k:prof(k)[0],bounds=(.01,4),method='bounded').x; rates.append(k);amps.append(prof(k)[1]);
print('exp rate quantiles',np.quantile(rates,[0,.1,.25,.5,.75,.9,1]));print('mean exp fit',np.mean([np.linalg.norm(np.c_[np.ones(len(times)),-np.expm1(-times*k)]@a-row) for k,a,row in zip(rates,amps,t)])); print('trajectory extremes by norm',t[np.argsort(np.linalg.norm(t,axis=1))[[0,12,36,60,71]]]); print('rate correlations',np.corrcoef(np.c_[rates,ai[:72],Y[:,0]].T)[0]);print('amp quantiles',np.quantile(amps,[0,.25,.5,.75,1],axis=0))
