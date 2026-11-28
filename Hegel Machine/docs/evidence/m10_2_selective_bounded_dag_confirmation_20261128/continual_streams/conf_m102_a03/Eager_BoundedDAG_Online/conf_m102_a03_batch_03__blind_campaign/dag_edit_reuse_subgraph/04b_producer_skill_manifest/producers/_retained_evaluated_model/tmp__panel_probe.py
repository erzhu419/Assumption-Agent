exec(open('/tmp/explore.py').read().split("for k in ('panel_a'")[0])
from scipy.optimize import least_squares
fs=np.array_split(np.random.default_rng(42).permutation(72),6)+[np.arange(52,72)]; out=np.c_[y,t]
def cv(b,alpha=0):
 ee=[]
 for q in fs:
  p=np.setdiff1d(np.arange(72),q); w=np.linalg.solve(b[p].T@b[p]+alpha*np.eye(b.shape[1]),b[p].T@out[p]) if alpha else np.linalg.lstsq(b[p],out[p],rcond=None)[0]; r=abs(b[q]@w-out[q]);ee.append([r[:,0].mean(),r[:,1:].mean(),r[:,-1].mean()])
 return np.max(ee,0),ee[-1]
for panel in ['panel_b','panel_c']:
 u=s[panel]['u']
 for ky,v in s[panel].items():
  if ky=='u':continue
  for name,feat in [('odd',(lambda z,w:np.c_[z,np.sin(z*w)])),('shifted',(lambda z,w:np.c_[aff(z),np.sin(z*w),np.cos(z*w)])),('sincos',(lambda z,w:np.c_[np.sin(z*w),np.cos(z*w)])),('sin',(lambda z,w:np.c_[aff(z),np.sin(z*w)])),('cos',(lambda z,w:np.c_[aff(z),np.cos(z*w)]))]:
   def coef(w):return np.linalg.lstsq(feat(u,w),v,rcond=None)[0]
   w=least_squares(lambda w:(feat(u,w)@coef(w)-v).ravel(),np.full(4,2.),bounds=(.2,6.)).x; co=coef(w); b=feat(u,w); lev=np.sum(np.linalg.svd(b,full_matrices=False)[0]**2,1); print(panel,ky,name,w,'rms',np.sqrt(np.mean((b@co-v)**2)),'loo',np.mean(abs((b@co-v)/(1-lev[:,None])))); z=feat(x,w)@co; a2=poly(z,2); a3=poly(z,3); bil=z[:,0]*z[:,2]+z[:,1]*z[:,3]; print(' cv',cv(np.c_[aff(z),bil]),cv(a2),cv(a3,.01),cv(np.c_[aff(x),a2[:,1:]],.01));
   if panel=='panel_b':print(' co',co.T)
for alpha in [.001,.01,.03,.1,.3,1]: print('cubic ridge',alpha,cv(poly(x,3),alpha))
