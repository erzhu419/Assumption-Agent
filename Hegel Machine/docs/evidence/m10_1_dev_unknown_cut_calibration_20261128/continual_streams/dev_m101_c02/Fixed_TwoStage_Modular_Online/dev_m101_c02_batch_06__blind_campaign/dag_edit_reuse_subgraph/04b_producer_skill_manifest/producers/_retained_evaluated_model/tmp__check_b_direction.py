import pickle,itertools,numpy as np
from scipy.optimize import least_squares,minimize_scalar
from panel_ab import fit_a,inv_a
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(p['panel_b'][k] for k in ('u','v')); x=p['train']['x']; c=fit_a(p['panel_a']); print('A coeff',c); t=inv_a(x,c); Y=p['train']['traj']; n=len(x)-20
np.set_printoptions(precision=4,suppress=True)
def aff(x):return np.c_[np.ones(len(x)),x]
def score(z):
 X=aff(z); return np.mean(abs(X[n:]@np.linalg.lstsq(X[:n],Y[:n],rcond=None)[0]-Y[n:]))
print('raw and A CV',score(x),score(t)); print('ranges B',u.min(0),u.max(0),v.min(0),v.max(0))
for tag, h, bd in [('sin',lambda x,f:np.sin(f*x),(1,4)),('rational',lambda x,f:x**3/(1+f*x*x),(.05,3)),('tanh',lambda x,f:np.tanh(f*x),(.2,3))]:
 for quadratic in (False,True):
  def feats(x,f):return np.c_[aff(x),h(x,f),x*np.roll(x,2,axis=1)[:,:] if quadratic else np.empty((len(x),0))]
  def fit(f):
   a=feats(u,f); C=np.linalg.lstsq(a,v,rcond=None)[0]; return np.linalg.norm(a@C-v),C
  f=minimize_scalar(lambda f:fit(f)[0],bounds=bd,method='bounded').x; e,C=fit(f)
  def pred(z):return feats(z,f)@C
  def inv(z):
   L=np.linalg.lstsq(aff(v),u,rcond=None)[0]; init=aff(z)@L
   res=[least_squares(lambda w:(pred(w[None,:])-row).ravel(),w0,max_nfev=100) for row,w0 in zip(z,init)]
   return np.array([r.x for r in res]),max(np.linalg.norm(r.fun) for r in res)
  ix,ex=inv(x); it,et=inv(t)
  print(tag,quadratic,round(f,4),'B rms',e/np.sqrt(v.size),'forward raw/A',score(pred(x)),score(pred(t)),'inverse raw/A',score(ix),score(it),'inverse residuals',ex,et,'A after B',score(inv_a(pred(x),c)))
  if tag=='sin' and not quadratic:
   print('B coeff',C); print('B residual svd',np.linalg.svd(pred(u)-v,compute_uv=False)); print('t vs u quantiles',*[np.quantile(a,[0,.1,.5,.9,1],axis=0) for a in (t,u,v)])
# Compare the dominant trajectory direction to additive and pairwise bases fitted only on training rows.
z=t; pair=list(itertools.combinations_with_replacement(range(4),2)); Q=np.c_[*[z[:,i]*z[:,j] for i,j in pair]]; print('trajectory harmonics')
for f in (.8,1.3,2.,2.4,2.8,3.2,3.6,4.,4.5,5.):
 for mode in (0,1,2):
  X=np.c_[aff(z),np.sin(f*z)]; X=np.c_[X,Q] if mode==1 else (np.c_[X,z*np.roll(z,-1,axis=1),z[:,0]*z[:,2],z[:,1]*z[:,3]] if mode==2 else X)
  w=np.linalg.lstsq(X[:n],Y[:n],rcond=None)[0]; print(f,mode,round(np.mean(abs(X[:n]@w-Y[:n])),5),round(np.mean(abs(X[n:]@w-Y[n:])),5),np.round(w[[*range(1,9)],-1],3))
