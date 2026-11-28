from probe import *
u,v=s['panel_b']['u'],s['panel_b']['v'];
def trial(f):
 d=ic(f(u)); c=ls(d,v); return np.sqrt(np.mean((d@c-v)**2))
for p in np.arange(1.5,8,.25):
 print('power',p,trial(lambda x:np.sign(x)*abs(x)**p)) if p in [2,3,4,5,6,7] else None
for k in range(3,16,2):
 d=ic(np.c_[*[u**p for p in range(1,k+1,2)]]); c=ls(d,v); h=(d@np.linalg.pinv(d)).diagonal(); print('odd',k,np.sqrt(np.mean((d@c-v)**2)),np.sqrt(np.mean(((d@c-v)/(1-h[:,None]))**2)),np.linalg.norm(c[1:].reshape(-1,16),axis=1))
for n,f in fs.items():
 if n in ['cos','square','cubic','exp']:continue
 for mode in range(2):
  def scalar(x,q):return (x if mode==0 else x**3)+q[0]*f(x*np.exp(q[1]))
  def err(q):
   d=ic(scalar(u,q));return (d@ls(d,v)-v).ravel()
  fit=min((least_squares(err,[a,b],max_nfev=180,bounds=([-100,-2.5],[100,2])) for a in [-2,-.5,.5,2] for b in [-.5,.5,1.1]),key=lambda v:v.cost)
  print('scalar',n,mode,np.sqrt(np.mean(fit.fun**2)),fit.x)
for q in [.22,1.25]:
 print('telemetry candidate',q,trial(lambda x:x**3/(1+q*x*x)),trial(lambda x:x*np.sqrt(q+x*x)),trial(lambda x:x*(1-np.exp(-q*x*x))))
