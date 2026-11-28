from screen import *
from scipy.optimize import least_squares
np.set_printoptions(precision=4,suppress=True)
u=s['panel_c']['u']; times=s['eval_times']; idx=np.arange(n)
def profile(D,V):return D@np.linalg.lstsq(D,V,rcond=None)[0]-V
for name,V in [('v1',s['panel_c']['v1']),('v2',s['panel_c']['v2'])]:
 for f in ([lambda q,p:aff(np.arctan(p[0]*q)),lambda q,p:np.c_[aff(q),np.arctan(p[0]*q)],lambda q,p:aff(np.arctan(q*p[:4]+p[4:])),lambda q,p:np.c_[aff(q),np.arctan(q*p[:4]+p[4:])],lambda q,p:np.c_[aff(np.arctan(p[0]*q)),q*np.roll(q,-1,axis=1)]] if name=='v1' else [lambda q,p:np.c_[aff(q),q*q/(1+p[0]*q*q)],lambda q,p:np.c_[aff(q),q*q/(1+p[:4]*q*q)],lambda q,p:np.c_[aff(q),np.sqrt(1+p[:4]*q*q)]]):
  o=least_squares(lambda p:profile(f(u,p),V).ravel(),np.r_[np.full(4,2. if name=='v1' else .2),np.zeros(4)],max_nfev=150);print(name,np.std(o.fun),o.x)
def shape(k):return -np.expm1(-k*times)
params=[]
for row in t:
 def prof(k):
  v=shape(k); a=v@row/(v@v); return a,np.mean((a*v-row)**2)
 opt=minimize_scalar(lambda k:prof(k)[1],bounds=(.01,5),method='bounded');params.append((prof(opt.x)[0],opt.x,prof(opt.x)[1]**.5))
params=np.array(params);print('single exponential first 20',params[:20], 'quantiles',np.quantile(params,[0,.1,.5,.9,1],axis=0));
for start in [[.4,1.4],[1.,2.],[1.,3.]]:
 o=least_squares(lambda k:profile(np.column_stack([shape(z) for z in k]),t.T).ravel(),start,bounds=(.001,10));D=np.column_stack([shape(z) for z in o.x]);print('double exponential',o.x,np.std(o.fun),np.linalg.lstsq(D,t.T,rcond=None)[0][:,:6]);
_,_,vt=np.linalg.svd(t,full_matrices=False);targets=np.c_[y,t@vt[:2].T,params[:,:2]]
for k,q in coords.items():
 print('corr',k,np.round(np.corrcoef(np.c_[q,targets].T)[:4,4:],2).T)
for target in [y,t@vt[0],t@vt[1]]:
 print('top univariate')
 fs=[]
 for k,q in coords.items():
  for j,z in enumerate(q.T):
   for name,z in [('linear',z),('cub',z**3),('sin',np.sin(2*z)),('atan',np.arctan(2*z)),('sq',z*z)]:fs.append((abs(np.corrcoef(z,target)[0,1]),k,j,name))
 print(sorted(fs,reverse=True)[:12])
