exec(open('/tmp/read13.py').read().split('from itertools import combinations_with_replacement as cr')[0])
from itertools import combinations_with_replacement as cr
from scipy.optimize import least_squares
h=np.column_stack([np.ones(len(q)),b]); y=tr['y']; r=y-h@ls(h,y)
folds=[np.arange(a,a+18) for a in (0,18,36,54)]+[np.arange(52,72)]
def cv(a):
 vals=[]
 for j in folds:
  i=np.setdiff1d(np.arange(len(q)),j); vals.append(np.mean(abs(a[j]@ls(a[i],y[i])-y[j])))
 return np.round(vals,5)
def prod(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
for label,a in [('q',q),('q2',q**2),('q3',q**3),('q4',q**4),('cos2',np.cos(2*p*q)),('cos3',np.cos(3*p*q)),('sin',np.sin(p*q)),('sin2',np.sin(2*p*q)),('diagb2',b*b),('b2',prod(b,b)),('q*b',prod(q,b)),('q2*b',prod(q*q,b)),('q*q',prod(q,q)),('b*tanhb',b*np.tanh(b)),('tanhb',np.tanh(b)),('b3',b**3),('q*sin',q*np.sin(p*q)),('q*cos',q*np.cos(p*q)),('tanhq',np.tanh(q)),('q2*cos',q*q*np.cos(p*q))]:
 ar=a-h@ls(h,a); corr=np.abs(r@ar)/(np.linalg.norm(ar,axis=0)*np.linalg.norm(r)); k=corr.argmax(); print('basis',label,k,round(corr[k],3),cv(np.c_[h,a[:,k]]),cv(np.c_[h,a]))
for typ in range(5):
 def gate(a):
  z=a[0]+q@a[1:5] if typ<3 else a[0]+b@a[1:5]
  if typ==0 or typ==3: return 1/(1+z*z)
  if typ==1 or typ==4: return np.tanh(z)
  return np.exp(-z*z)
 for augment in (False, True):
  def design(a):
   g=gate(a); return np.c_[h,g[:,None]*h] if augment else np.c_[np.ones(len(q)),g[:,None]*h]
  out=[]; rng=np.random.default_rng(0)
  for j in folds:
   i=np.setdiff1d(np.arange(len(q)),j)
   def residual(a):
    f=design(a)[i]; return f@ls(f,y[i])-y[i]
   starts=np.r_[np.c_[np.zeros(4),np.eye(4)*.47],rng.normal(size=(7,5))*.5]; best=min((least_squares(residual,x,bounds=(-6,6),max_nfev=120) for x in starts),key=lambda x:x.cost); f=design(best.x); out.append((round(np.mean(abs(f[j]@ls(f[i],y[i])-y[j])),5),round(np.sqrt(2*best.cost/len(i)),5),np.round(best.x,2).tolist()))
  print('profile',typ,augment,out)
