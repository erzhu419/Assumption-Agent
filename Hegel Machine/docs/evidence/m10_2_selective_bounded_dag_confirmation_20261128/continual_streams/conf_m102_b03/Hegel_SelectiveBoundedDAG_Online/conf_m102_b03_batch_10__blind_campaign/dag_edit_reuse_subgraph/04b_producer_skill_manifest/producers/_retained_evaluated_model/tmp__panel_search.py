import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
forms={'softsign':lambda w:w/(1+abs(w)),'tanh':np.tanh,'sin':np.sin,'cos':np.cos,'rational2':lambda w:w*w/(1+w*w),'rational1':lambda w:w/(1+w*w),'atan':np.arctan,'smoothsign':lambda w:w/np.sqrt(1+w*w),'gaussian':lambda w:np.exp(-w*w),'signed_square':lambda w:w*abs(w),'square':lambda w:w*w,'cube':lambda w:w**3}
for pk,vk in [('panel_a','v'),('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=[s[pk][k] for k in ('u',vk)]; print(pk,vk,'affine RMSE', np.sqrt(np.mean((np.c_[np.ones(len(u)),u]@ls(np.c_[np.ones(len(u)),u],v)-v)**2)),flush=True)
 scores=[]
 for name,fun in forms.items():
  def design(p): return np.c_[np.ones(len(u)),u,fun(p[0]*u+p[1:])]
  def resid(p):
   d=design(p); return (d@ls(d,v)-v).ravel()
  fits=[least_squares(resid,[a,0,0,0,0],bounds=([.03]+[-2]*4,[5]+[2]*4),max_nfev=90) for a in [.6,1.3,2.2]]
  f=min(fits,key=lambda f:f.cost); scores.append((np.sqrt(np.mean(f.fun**2)),name,np.round(f.x,4).tolist()))
 print(*sorted(scores)[:8],sep='\n',flush=True)
 if pk=='panel_a':
  for j in range(4):
   o=np.argsort(u[:,j]);print('diag',j,np.round(np.c_[u[o,j],v[o,j]][::4],3).tolist())
