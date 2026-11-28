from panels import *
from scipy.optimize import least_squares
p=s['panel_a']; u=p['u']; v=p['v']; A,_=poly(u,2); _,e=robust(A,v); good=np.argsort(np.linalg.norm(e,axis=1))[:-3]; A,n=poly(u,3); b=np.linalg.lstsq(A[good],v[good],rcond=None)[0]; print('A clean cubic MAE',np.mean(np.abs(A[good]@b-v[good]))); print([(k,t.round(5).tolist()) for k,t in zip(n,b) if np.max(np.abs(t))>.001]);
p=s['panel_c']; u=p['u']; v=p['v2'];
for kind,fn in {'rat':lambda x,a:x*x/(1+a*x*x),'cos':lambda x,a:(1-np.cos(a*x))/a**2,'gauss':lambda x,a:-np.expm1(-a*x*x)/a,'logcosh':lambda x,a:np.log(np.cosh(a*x))/a**2,'tanh2':lambda x,a:np.tanh(a*x*x)/a,'soft':lambda x,a:(np.sqrt(1+a*x*x)-1)/a,'sin2':lambda x,a:np.sin(a*x*x)/a,'poly4':lambda x,a:x*x-a*x**4}.items():
 for tie in (False,True):
  def fit(q,ret=False):
   a=np.exp(q[0]); f=fn(u,a); M=np.c_[np.ones(len(u)),u+q[1]*f] if tie else np.c_[np.ones(len(u)),u,f]; B=np.linalg.lstsq(M,v,rcond=None)[0]; E=M@B-v; return (B,E) if ret else E.ravel()
  opt=least_squares(fit,[-1,.8] if tie else [-1],bounds=([-7,-5] if tie else [-7],[2,5] if tie else [2])); B,E=fit(opt.x,True); print(kind,tie,round(np.exp(opt.x[0]),6),opt.x[1:].round(5),round(np.mean(np.abs(E)),7))
print('B coefficients',np.linalg.lstsq(np.c_[np.ones(24),s['panel_b']['u']],s['panel_b']['v'],rcond=None)[0]); print('traj SVD time vectors',np.linalg.svd(s['train']['traj'],full_matrices=False)[2][:2])
