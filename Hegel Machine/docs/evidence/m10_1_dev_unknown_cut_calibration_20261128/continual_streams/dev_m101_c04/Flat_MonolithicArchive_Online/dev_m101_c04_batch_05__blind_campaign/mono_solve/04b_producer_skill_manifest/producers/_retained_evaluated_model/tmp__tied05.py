import numpy as np,pickle,importlib.util
from scipy.optimize import least_squares
p='/tmp/hegel_continual_archive/dev_m101_c04_batch_03_solver.py'; sp=importlib.util.spec_from_file_location('arch',p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; y=t['y']; ls=m._ls; aff=m._affine
a,g=m._calibrate(s['panel_a']); u=(t['x']-a)/g; z=m._operator(s['panel_b'])(u); rng=np.random.default_rng(912); folds=[np.arange(72)]+[rng.permutation(72) for _ in range(5)]; folds.append(np.argsort(np.linalg.norm(z,axis=1)))
np.set_printoptions(precision=5,suppress=True)
c=s['panel_c']; print('times',s['eval_times'],'telemetry',s['telemetry_vec']);
for k in ['v1','v2']:
 f=aff(c['u']); print(k,'affine rmse',np.std(c[k]-f@ls(f,c[k])), 'scale',np.std(c[k]),'coefs',ls(f,c[k]).round(3))
A=aff(z); cross=np.c_[z[:,0]*z[:,2],z[:,1]*z[:,3]]; modes={'tanh':np.tanh,'atan':np.arctan,'asinh':np.arcsinh,'rational':lambda w:w/(1+w*w),'softsign':lambda w:w/np.sqrt(1+w*w),'erf':__import__('scipy.special',fromlist=['erf']).erf,'sin':np.sin,'rational3':lambda w:w**3/(1+w*w),'log':lambda w:np.sign(w)*np.log1p(abs(w)),'exp':lambda w:np.sign(w)*-np.expm1(-abs(w))}
for mode,h in modes.items():
 for tied_cross in [True,False]:
  C=cross.sum(1,keepdims=True) if tied_cross else cross; B=aff(C)
  def pred(tr,ret=False):
   def des(p):
    w=A@p; return np.c_[B,w,h(w)]
   def r(p):
    D=des(p)[tr]; return D@ls(D,y[tr])-y[tr]
   w=ls(np.c_[A,C][tr],y[tr])[:5]; w[0]=0
   opt=min((least_squares(r,w*k,bounds=(-15,15),max_nfev=120) for k in [.65,1.4,2.3]),key=lambda o:o.cost); D=des(opt.x); b=ls(D[tr],y[tr]); pred=D@b
   return (np.std(pred-y),opt.x,b) if ret else pred
  print(mode,tied_cross,[round(abs(pred(i[:56])[i[56:]]-y[i[56:]]).mean(),6) for i in folds],pred(np.arange(72),True))
# Check whether a single index applied to the affine-plus-cross expression accounts for remaining residuals.
for mode,h in {k:modes[k] for k in ['tanh','atan','rational','softsign','erf','sin']}.items():
 for vary in [0,1,2]:
  X=np.c_[A,cross.sum(1)] if vary else np.c_[A,cross]; B=aff(cross.sum(1)) if vary==2 else np.ones((len(y),1))
  def pred(tr,ret=False):
   def des(p):
    w=X@p;return np.c_[B,w,h(w)]
   def r(p):
    D=des(p)[tr];return D@ls(D,y[tr])-y[tr]
   init=ls(X[tr],y[tr]); opt=min((least_squares(r,init*k,max_nfev=150,bounds=(-15,15)) for k in [.55,1.2,2.3]),key=lambda o:o.cost); D=des(opt.x); b=ls(D[tr],y[tr]); prediction=D@b
   return (np.std(prediction-y),opt.x,b) if ret else prediction
  print('inside',mode,vary,[round(abs(pred(i[:56])[i[56:]]-y[i[56:]]).mean(),6) for i in folds],pred(np.arange(72),True))
