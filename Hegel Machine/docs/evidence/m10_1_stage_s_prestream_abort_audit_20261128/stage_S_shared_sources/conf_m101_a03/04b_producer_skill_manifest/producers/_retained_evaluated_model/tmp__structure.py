import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']; np.set_printoptions(precision=6,suppress=True,linewidth=130)
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def quad(x):return np.c_[aff(x),*[x[:,i]*x[:,j] for i in range(4) for j in range(i,4)]]
a=s['panel_a']; A=aff(a['u']); v=a['v']; M=ls(A,v)
for _ in range(8):
 r=np.linalg.norm(A@M-v,axis=1); keep=np.argsort(r)[:41]; M=ls(A[keep],v[keep])
r=np.linalg.norm(A@M-v,axis=1); keep=r<max(.001,5*np.median(r)); M=ls(A[keep],v[keep]); b=s['panel_b']; N=ls(aff(b['u']),b['v']); print('times',t,'traj sv',np.linalg.svd(z,compute_uv=False),'A/B',M,N,'A inliers',sum(keep),sep='\n')
def dump(name,w):
 c=ls(quad(w),y); H=np.zeros((4,4)); H[np.triu_indices(4)]=c[5:];H=(H+H.T)/2; print(name,'lin',c[:5],'H',H,'eig',np.linalg.eigh(H)[0],sep='\n')
for name,w in [('raw',x),('a',aff(x)@M),('a_inv',(x-M[0])@np.linalg.inv(M[1:])),('b',aff(x)@N),('ab',aff(aff(x)@M)@N),('a_inv_b',aff((x-M[0])@np.linalg.inv(M[1:]))@N)]:dump(name,w)
C=ls(aff(x),z);print('raw traj coeff',C,'singular',np.linalg.svd(C,compute_uv=False),sep='\n')
for name,fun,ini in [('exp+t',lambda p: np.c_[-np.expm1(-p[0]*t),t],[1]),('two-exp',lambda p: np.c_[-np.expm1(-p[0]*t),-np.expm1(-p[1]*t)],[.3,1.6]),('two-exp+t',lambda p: np.c_[-np.expm1(-p[0]*t),-np.expm1(-p[1]*t),t],[.3,1.6]),('exp+t+sine',lambda p: np.c_[-np.expm1(-p[0]*t),t,np.sin(p[1]*t)],[1,.6]),('exp+sin',lambda p: np.c_[-np.expm1(-p[0]*t),np.sin(p[1]*t)],[1,.6])]:
 def err(p):
  T=fun(p); return (C-ls(T,C.T).T@T.T).ravel()
 opt=least_squares(err,ini,bounds=(.001,15)); print(name,opt.x,'rms',np.sqrt(np.mean(opt.fun**2)))
print('response quadratic residual rms',np.sqrt(np.mean((y-quad(x)@ls(quad(x),y))**2)))
