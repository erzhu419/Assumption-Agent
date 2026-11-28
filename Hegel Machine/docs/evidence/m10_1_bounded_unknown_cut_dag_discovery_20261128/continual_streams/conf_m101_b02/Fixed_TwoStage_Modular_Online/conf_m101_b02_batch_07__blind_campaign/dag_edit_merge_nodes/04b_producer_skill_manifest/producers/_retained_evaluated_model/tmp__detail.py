exec(open('/tmp/structure_fixed.py').read().split('for mode')[0])
from scipy.linalg import svd
for k in s:
 print(k, list(s[k]) if isinstance(s[k],dict) else np.asarray(s[k]).shape)
print('data',n,np.round(np.column_stack((x,y,T[:,0],T[:,-1]))[:12],3));print('T svd',svd(T,compute_uv=False));print('centered trajectory',svd(T-T.mean(0),compute_uv=False))
# Profile a shared relaxation rate, then inspect left-over time modes and their relationship to the scalar.
def cur(k):return np.column_stack((np.ones(len(t)),-np.expm1(-k*t)))
def er(k):
 d=cur(k);return np.mean((T-(d@fit(d,T.T)).T)**2)
k=minimize_scalar(er,bounds=(.1,1.2),method='bounded').x;d=cur(k);P=fit(d,T.T).T;R=T-P@d.T;U,G,V=svd(R,full_matrices=False);print('time profile',k,er(k)**.5,'scores',G); print('correlations',np.corrcoef(np.column_stack((y,P,U[:,:3],x)).T)[:3].round(3))
# Fit B's mapped odd component together with a small shared even component to identify the calibration residual.
xx=B['u']; yy=B['v'];
for name,fn in [('sine',lambda z,p:z+p[1]*np.sin(p[0]*z)),('exp',lambda z,p:z+p[1]*z*np.exp(-p[0]*z*z)),('rat',lambda z,p:z+p[1]*z/(1+p[0]*z*z))]:
 for mode in range(4):
  def D(z,p):
   O=fn(z,p[:2]); E=[[],[z*z],[z*z/(1+p[2]*z*z)],[np.sqrt(1+p[2]*z*z)-1]][mode];return aff(np.column_stack([O,*E]))
  def res(p):
   d=D(xx,p);return (d@fit(d,yy)-yy).ravel()
  sol=least_squares(res,[2.8 if name=='sine' else 1,-.4 if name=='sine' else -1]+([.8] if mode>=2 else []),bounds=([.001,-3]+([.001] if mode>=2 else []),[6,0]+([5] if mode>=2 else []))); W=fit(D(xx,sol.x),yy);print('B-even',name,mode,sol.x.round(4),np.std(res(sol.x)),np.linalg.svd(res(sol.x).reshape(-1,4),compute_uv=False).round(4));
