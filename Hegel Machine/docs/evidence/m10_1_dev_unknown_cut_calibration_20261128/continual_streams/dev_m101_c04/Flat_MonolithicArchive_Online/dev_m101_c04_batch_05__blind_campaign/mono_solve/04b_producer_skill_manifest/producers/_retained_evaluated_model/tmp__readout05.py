import numpy as np,pickle,importlib.util,itertools as it
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import lasso_path
p='/tmp/hegel_continual_archive/dev_m101_c04_batch_03_solver.py'; sp=importlib.util.spec_from_file_location('arch',p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; y=t['y']; times=s['eval_times']; ls=m._ls; aff=m._affine
np.set_printoptions(precision=6,suppress=True)
a,g=m._calibrate(s['panel_a']); u=(t['x']-a)/g; op=m._operator(s['panel_b']); print('cal',a,g); print('panel B residual std',np.std(op(s['panel_b']['u'])-s['panel_b']['v'])); z=op(u); n=len(y); rng=np.random.default_rng(912); folds=[np.arange(n)]+[rng.permutation(n) for _ in range(5)]
poly=PolynomialFeatures(3,include_bias=False); F=poly.fit_transform(z); names=poly.get_feature_names_out(); print('full cubic coefficients',[(k,round(c,4)) for k,c in zip(names,ls(aff(F),y)[1:])]); print('full cubic residual',np.std(y-aff(F)@ls(aff(F),y)))
for basis,q in [('z',z),('u',u)]:
 poly=PolynomialFeatures(3,include_bias=False); F=poly.fit_transform(q); names=poly.get_feature_names_out()
 for alpha in [.03,.01,.003,.001,.0003]:
  scores=[]
  for ix in folds+[np.arange(n)]:
   tr,te=ix[:56],ix[56:]; tr=ix if len(scores)==len(folds) else tr
   Fa=F[tr]; scale=Fa.std(0); _,cs,_=lasso_path((Fa-Fa.mean(0))/scale,y[tr]-y[tr].mean(),alphas=[alpha],max_iter=20000); keep=np.flatnonzero(abs(cs[:,0])>1e-7); D=aff(F[:,keep]); c=ls(D[tr],y[tr]); scores.append(abs(D[te]@c-y[te]).mean())
  print(basis,alpha,np.round(scores[:6],5),'resid',round(np.std(y-D@c),5),'terms',[(k,round(v,4)) for k,v in zip(names[keep],c[1:])]);
base=aff(z); pairs=[(i,j) for i in range(4) for j in range(i,4)]; base=np.c_[base,z[:,0]*z[:,2],z[:,1]*z[:,3]]
shapes={}
for i,j in pairs:
 for sign in ([1,-1] if i!=j else [1]):
  v=z[:,i] if i==j else z[:,i]+sign*z[:,j]; shapes[f'z{i} {sign} z{j}']=v
for i in range(4): shapes[f'u{i}']=u[:,i]
for k,fun in {'cub':lambda v:v**3,'signedsq':lambda v:v*abs(v),'tanh':lambda v:np.tanh(1.7*v),'sin':lambda v:np.sin(1.7*v),'sat':lambda v:v/(1+1.7*v*v)}.items():
 rr=[]
 for name,v in shapes.items():
  D=np.c_[base,fun(v)]; rr.append((np.std(y-D@ls(D,y)),name))
 print('one shape',k,sorted(rr)[:4]);
traj=t['traj']; coeff=ls(aff(z),traj); print('trajectory projection SVD',np.linalg.svd(aff(z)@coeff,compute_uv=False)); print('orthogonal residual std',np.std(traj-aff(z)@coeff))
for name,b in [('independent',lambda p:np.eye(len(times))),('oneexp',lambda p:-np.expm1(-times[:,None]*p[:1])),('twoexp',lambda p:-np.expm1(-times[:,None]*p)),('exp_sine',lambda p:np.c_[-np.expm1(-times*p[0]),np.sin(times*p[1])]),('exp_lin',lambda p:np.c_[-np.expm1(-times*p[0]),times])]:
 ss=[]
 for ix in folds:
  tr,te=ix[:56],ix[56:]; A=aff(z); target=(np.linalg.qr(A[tr])[0].T@traj[tr]).T
  def fun(p):
   h=b(p); return (h@ls(h,target)-target).ravel()
  p=least_squares(fun,[.4,1.7],bounds=(.02,12.)).x if name!='independent' else []; h=b(p); pred=A[te]@ls(A[tr],ls(h,traj[tr].T).T)@h.T; ss.append(abs(pred-traj[te]).mean())
 print(name,np.round(ss,6),'params',p)
