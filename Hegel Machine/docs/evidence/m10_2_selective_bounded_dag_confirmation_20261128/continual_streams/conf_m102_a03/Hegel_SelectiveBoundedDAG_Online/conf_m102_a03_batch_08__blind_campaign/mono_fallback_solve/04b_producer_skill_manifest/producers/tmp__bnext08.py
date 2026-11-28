import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as cwr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=[s['panel_b'][k] for k in ('u','v')]; d=np.load('/tmp/feat08.npz'); z=d['z']; fit=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-11)[0]
for fn in [np.sin,np.tanh,np.arctan,lambda x:x/(1+x*x),lambda x:x*np.exp(-x*x)]:
 for mod in range(8):
  def m(x,h):
   k=h[0]; g=fn(k*x); a=x+h[1]*g; ex=[a+h[2]*x**3,a+h[2]*x**2,a+h[2]*x*np.roll(x,-1,axis=1),a+h[2]*x*np.roll(x,1,axis=1),a+h[2]*np.sin(2*k*x),a+h[2]*np.sin(h[3]*x),a+h[2]*x/(1+h[3]**2*x*x),a+h[2]*np.sin(h[3]*x+h[4])][mod];return np.c_[np.ones(len(x)),ex]
  def r(h):
   A=m(u,h);return (A@fit(A,v)-v).ravel()
  results=[least_squares(r,[f,-.4,0,b,0],bounds=([.1,-5,-5,.1,-3],[8,5,5,8,3]),max_nfev=200) for f in (1.1,2.6,3.7) for b in (1.5,3.2,5.)]; o=min(results,key=lambda t:t.cost); rms=np.std(o.fun)
  print(fn.__name__,mod,round(rms,7),o.x.round(4));
  if rms<.001: np.save('/tmp/bexact08.npy',m(z,o.x)@fit(m(u,o.x),v))
