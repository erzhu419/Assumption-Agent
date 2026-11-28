import pickle,numpy as np
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; X=c['true_channel_loads']; Y=c['bridge_raw']; S=c['single_channel_sweeps']; k=(abs(Y)<5).all(1); X,Y=X[k],Y[k]; ok=abs(S['cmd_stress'])<5;j=S['lab_id'][ok];z=S['true_stress'][ok];v=S['cmd_stress'][ok]; W=X[:,[2,3,0,1]]
def axis(f):
 A=np.column_stack([np.ones(len(z)),z,np.sin(f*z)]); b=np.stack([np.linalg.lstsq(A[j==q],v[j==q],rcond=None)[0] for q in range(4)]); return (A*b[j]).sum(1)-v,b
f=least_squares(lambda p:axis(p[0])[0],[2.5]).x[0]; b=axis(f)[1]; R=Y-b[:,0]-X*b[:,1]-np.sin(f*X)*b[:,2]
forms={
 'rational square':lambda x,a:x*x/(1+a*x*x), 'rational abs':lambda x,a:x*x/(1+a*abs(x)), 'rational fourth':lambda x,a:x*x/(1+a*x**4),
 'tanh square':lambda x,a:np.tanh(a*x*x),'square tanh':lambda x,a:x*np.tanh(a*x),'cos':lambda x,a:1-np.cos(a*x),'cos square':lambda x,a:(1-np.cos(a*x))/(1+x*x), 'exp':lambda x,a:-np.expm1(-a*x*x), 'log':lambda x,a:np.log1p(a*x*x),'sqrt':lambda x,a:np.sqrt(1+a*x*x)-1,'rational free power':lambda x,a,p:x*x/(1+a*abs(x)**p), 'squared sine':lambda x,a:np.sin(a*x)**2}
folds=np.array_split(np.random.default_rng(42).permutation(len(X)),5)
for name,g in forms.items():
 def estimate(fit):
  def inner(p,idx,ret=False):
   G=g(W,*p); obs=[]; coeff=[]
   for q in range(4):
    A=np.column_stack([W[:,q],G[:,q]]); t=np.linalg.lstsq(A[fit],R[fit,q],rcond=None)[0]; obs.append(A[idx]@t-R[idx,q]); coeff.append(t)
   return (np.array(obs).T,np.array(coeff)) if ret else np.ravel(obs)
  p=least_squares(lambda p:inner(p,fit), [.7, 2.] if name.endswith('free power') else [.7],bounds=(.005,5)).x
  return p,inner
 e=np.zeros_like(R)
 for ids in folds:
  p,fn=estimate(np.setdiff1d(np.arange(len(X)),ids)); e[ids],_=fn(p,ids,True)
 p,fn=estimate(np.arange(len(X))); _,co=fn(p,np.arange(len(X)),True)
 print(name,p,np.quantile(abs(e),[.5,.9,1]),'worstfold',max(np.quantile(abs(e[i]),.9) for i in folds),'normalized coeff',np.round(co/b[:,1,None],5).tolist())
