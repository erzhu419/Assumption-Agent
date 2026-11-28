from refine import *
from scipy.optimize import least_squares

def investigate():
 z=calibrate()(s['train']['x']); y=s['train']['y']; Y=s['train']['traj']; t=s['eval_times']; u,v=s['panel_b']['u'],s['panel_b']['v'];
 for kind in ['rational','tanh']:
  e,p,op,g=fitop(u,v,kind); w=op(z); opp=(w[:,:2]*w[:,2:]).sum(1); H=np.c_[aff(w),opp]; B=H[:52]; b=H[52:]; idx=np.argsort(np.linalg.norm(z[52:],axis=1))[-5:];
  fit=L(B,y[:52],rcond=None)[0];print(kind,'linear',fit,'yMAE',np.mean(abs(b@fit-y[52:])), 'tail',np.mean(abs(b@fit-y[52:])[idx]));
  for name,link in [('tanh',np.tanh),('rational',lambda a:a/(1+a*a)),('cubic',lambda a:a**3),('atan',np.arctan),('sin',np.sin)]:
   def d(h,p):
    a=h@p;return np.c_[np.ones(len(a)),a,link(a)]
   def res(p):
    m=d(B,p);return m@L(m,y[:52],rcond=None)[0]-y[:52]
   q=min((least_squares(res,fit*c,max_nfev=100) for c in [.15,.4,.8,1.25]),key=lambda q:q.cost); m=d(B,q.x); coef=L(m,y[:52],rcond=None)[0]; err=abs(d(b,q.x)@coef-y[52:]);print(name,'rms','%.5f'%np.sqrt(np.mean(res(q.x)**2)),'cv','%.5f'%err.mean(),'tail','%.5f'%err[idx].mean(),'coef',coef,'index',q.x)
  C=L(B,Y[:52],rcond=None)[0];print('temporal coef',C)
  for name,M,N in [('aff',B[:,:5],b[:,:5]),('paired',B,b),('quad',mon(w[:52],2),mon(w[52:],2))]:
   def mode(q):return -np.expm1(-np.exp(q[:,None])*t)
   F=L(M,Y[:52],rcond=None)[0];
   def f(q):
    m=mode(q);return L(m.T,F.T,rcond=None)[0].T@m
   q=least_squares(lambda p:(M@(f(p)-F)).ravel(),np.log([.4,1.8]),bounds=(np.log([.005,.25]),np.log([2,15]))).x; err=abs(N@f(q)-Y[52:]); print('state',name,np.exp(q),err.mean(),err.mean(0).max(),err[idx].mean(),np.sqrt(np.mean((M@f(q)-Y[:52])**2)))
if __name__=='__main__': investigate()
