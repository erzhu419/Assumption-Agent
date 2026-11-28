exec(open('/tmp/operators.py').read().split('res=[]')[0])
from scipy.optimize import least_squares
from scipy.special import erf
for label,obs in [('C1',V),('C2',W)]:
 for name,g in [('tanh',np.tanh),('sin',np.sin),('atan',np.arctan),('soft',lambda z:z/np.hypot(1,z)),('rat',lambda z:z/(1+z*z)),('tanh2',lambda z:np.tanh(z)**2),('softabs',lambda z:np.hypot(1,z)),('exp',lambda z:np.exp(-z*z)),('cubic',lambda z:z**3)]:
  def f(z,h): return np.c_[aff(z),g(z*h[:4]+h[4:]),z*z] if label=='C2' else np.c_[aff(z),g(z*h[:4]+h[4:])]
  def resid(h):
   p=f(u,h); return (obs-p@np.linalg.lstsq(p,obs,rcond=1e-9)[0]).ravel()
  r=least_squares(resid,np.r_[np.full(4,2 if label=='C1' else 1),np.zeros(4)],bounds=(np.r_[np.full(4,.1),np.full(4,-2)],np.r_[np.full(4,5),np.full(4,2)]),max_nfev=140)
  print(label,name,round(np.std(r.fun),7),np.round(r.x,3),round(score(lambda z:f(z,r.x),obs),5))
 print(label,'powers',[(d,round(score(lambda z:np.c_[aff(z),*[z**k for k in range(2,d+1)]],obs),6)) for d in range(2,5)])
Q=np.c_[aff(x),mono(x,2)]; print('telemetry',s['telemetry_vec'],'summary',s['train']['summary']); print('empirical',Y[:,0].mean(),Y[:,0].std(),Y[:,-1].mean(),Y[:,-1].std())
for name,z in [('x',x),('A',A(x)),('AB',B(A(x))),('B',B(x)),('AiB',B(Ai(x)))]:
 print(name)
 G=np.c_[aff(z),mono(z,2)]; c=np.linalg.lstsq(G,Y[:,[0,-1]],rcond=None)[0]; print(np.round(c.T,3));
