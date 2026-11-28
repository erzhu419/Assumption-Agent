import numpy as np,pickle
from scipy.optimize import least_squares
from numpy.polynomial.chebyshev import chebvander
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; X,y,T=(tr[k] for k in ('x','y','traj')); times=s['eval_times']
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-11)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
a,b=s['panel_a'],s['panel_b']; keep=np.max(abs(a['v']-a['u']),axis=1)<1
u,v=a['u'][keep],a['v'][keep]; g=np.sum((u-u.mean(0))*(v-v.mean(0)),axis=0)/np.sum((u-u.mean(0))**2,axis=0); off=(v-u*g).mean(0); X=(X-off)/g
U,V=b['u'],b['v']; np.set_printoptions(precision=5,suppress=True); print('calibration',g,off,'range latent',X.min(0),X.max(0)); print('B coefficient degree 1 and 3',ls(np.c_[aff(U),U**3],V)); print('SVD linear and cubic coefficients',np.linalg.svd(ls(np.c_[aff(U),U**3],V)[1:],compute_uv=False)); print('residual degree 3',np.sqrt(np.mean((np.c_[aff(U),U**3]@ls(np.c_[aff(U),U**3],V)-V)**2)))

def zs(x,p,mode):
 d,l,a,f,c,h=p
 if mode=='cubic': z=x**3/(1+d*x*x)+l*x+a*np.sin(f*x)+c*np.sin(h*x)
 elif mode=='linear': z=x/(1+d*x*x)+l*x+a*np.sin(f*x)+c*np.sin(h*x)
 elif mode=='tanh': z=np.tanh(d*x)+l*x+a*np.sin(f*x)+c*np.sin(h*x)
 elif mode=='sin': z=np.sin(d*x)+l*x+a*np.sin(f*x)+c*np.sin(h*x)
 elif mode=='oddpoly': z=x**3+l*x+a*x**5+c*x**7
 return aff(z)

def scalar(z):return np.c_[z,z[:,1]*z[:,3]+z[:,2]*z[:,4]]
def run(mode):
 def profile(p,train=np.arange(56),test=np.arange(56,72)):
  a,x,q=[zs(v,p,mode) for v in (U,X[train],X[test])]; op=ls(a,V); z=scalar(np.c_[np.ones(len(x)),x@op]); zz=scalar(np.c_[np.ones(len(q)),q@op]); cy,ct=ls(z,y[train]),ls(x,T[train]); return np.r_[(a@op-V).ravel(), .6*(z@cy-y[train]),(.6/np.sqrt(T.shape[1]))*(x@ct-T[train]).ravel()], zz@cy,q@ct, np.sqrt(np.mean((a@op-V)**2))
 limits=([.001,-5,-2,.1,-2,.1],[10,5,2,35,2,35]); sols=[]
 for f,h in [(2,7),(5,10),(7,12),(9,14),(11,16),(13,18),(15,20),(17,24),(20,27),(24,30),(7,20),(12,23)]:
  p=least_squares(lambda p:profile(p)[0],[.7,-.05,.02,f,.006,h],bounds=limits,max_nfev=160).x
  r,yp,tp,panel=profile(p); sols.append((np.linalg.norm(r),np.mean(abs(yp-y[56:])),np.mean(abs(tp-T[56:])),panel,p))
 print(mode, sorted(sols,key=lambda x:x[0])[:3],flush=True)
for mode in ['cubic','linear','tanh','sin','oddpoly']:run(mode)
