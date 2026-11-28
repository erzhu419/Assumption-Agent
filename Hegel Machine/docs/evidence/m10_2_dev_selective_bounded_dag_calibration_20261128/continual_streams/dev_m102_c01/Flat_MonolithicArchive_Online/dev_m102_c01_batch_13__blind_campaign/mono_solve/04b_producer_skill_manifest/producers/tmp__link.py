exec(open('/tmp/probe.py').read().split('ops={}')[0]); from scipy.optimize import least_squares
from itertools import combinations_with_replacement
F=np.column_stack([np.ones(len(x))]+[np.prod(x[:,ii],axis=1) for d in (1,2) for ii in combinations_with_replacement(range(4),d)]); n=52
print('times',t); print('stats',np.min(y),np.max(y),np.min(T[:,-1]),np.max(T[:,-1]))
p=np.linalg.svd(T[:n],full_matrices=False)[2][0]; p*=np.sign(p[-1]); A=(T@p)*p[-1]; print('corr',np.corrcoef(y,A)[0,1]); print('y-state plots',np.c_[y,A][np.argsort(y)][::6]);
for out in [y,A]:
 m=out[:n].mean(); sd=out[:n].std(); Y=(out-m)/sd
 for mode in ['lin','tanh','sin','asinh','atan','sinh','soft','poly','rational','quadplus']:
  def pred(w):
   h=F@w[:15]; k=w[-2]; b=w[-1]
   if mode=='lin': return h
   if mode=='tanh':return b+k*np.tanh(h/k)
   if mode=='sin':return b+k*np.sin(h/k)
   if mode=='asinh':return b+k*np.arcsinh(h/k)
   if mode=='atan':return b+k*np.arctan(h/k)
   if mode=='sinh':return b+k*np.sinh(np.clip(h/k,-6,6))
   if mode=='soft':return b+h/(1+abs(h)/k)
   if mode=='poly':return b+h+k*h**3
   if mode=='quadplus':return b+h+k*h*h
   return b+h/(1+(h/k)**2)
  options=[]
  for k in (1.,2.,4.,8.):
   w=least_squares(lambda w:(pred(w)-Y)[:n],np.r_[fit(F[:n],Y[:n]),k,0.],max_nfev=150).x; e=(pred(w)-Y)*sd; options.append((np.mean(abs(e[:n])),np.mean(abs(e[n:])),w[-2:].round(3)))
  print(mode,min(options,key=lambda r:r[0]))
