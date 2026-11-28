exec(open('/tmp/probe_panels.py').read().split('for k,panel')[0])
from scipy.optimize import least_squares
T0=T@np.linalg.svd(T,full_matrices=False)[2][:3].T
b=s['panel_b']; c=s['panel_c']
sets=[('B',b['u'],b['v']),('C1',c['u'],c['v1']),('C2',c['u'],c['v2']),('C12',c['v1'],c['v2']),('train',u,np.c_[y,T0[:,0]/1.8])]
def err(X,Y,it,iv):
 co=reg(X[it],Y[it]); return np.mean(abs(X[iv]@co-Y[iv]),axis=0)
def f(x,k,fam):
 if fam=='tanh':return np.tanh(k*x)
 if fam=='sin':return np.sin(k*x)
 if fam=='rat':return x/(1+k*x*x)
 if fam=='asinh':return np.arcsinh(k*x)
 if fam=='sinh':return np.sinh(k*x)
 if fam=='atan':return np.arctan(k*x)
 if fam=='sat':return x/np.sqrt(1+k*x*x)
 if fam=='inv3':
  q=x.copy()
  for _ in range(10): q-=(q+k*q**3-x)/(1+3*k*q*q)
  return q
 if fam=='cube':return x**3
 if fam=='quint':return x**5
 if fam=='signed2':return x*abs(x)
 if fam=='signed05':return np.sign(x)*np.sqrt(abs(x))
for name,q,Y in sets:
 print('\n',name); N=len(q); folds=np.array_split(np.random.default_rng(133).permutation(N),6); allidx=np.arange(N); scores1=[]
 for fam in ('tanh','sin','rat','asinh','sinh','atan','sat','inv3','cube','quint','signed2','signed05'):
  for k in ([.2,.5,.8,1,1.25,1.5,2,2.5,3,4,5,7] if fam not in ('cube','quint','signed2','signed05') else [1]):
   w=f(q,k,fam)
   for mode,X in [('phi',poly(w,1)),('phiq',np.c_[poly(w,1),q*q]),('phi2',poly(w,2)),('xphi',np.c_[poly(q,1),w]),('xpq',np.c_[poly(q,1),q*q,w]),('x2phi',np.c_[poly(q,2),w])]:
    if X.shape[1]>=N-4:continue
    sl=np.mean([err(X,Y,np.setdiff1d(allidx,iv),iv) for iv in folds],axis=0); scores1.append((float(sl.mean()),fam,k,mode,sl, mse(X,Y)))
 print(*sorted(scores1,key=lambda v:v[0])[:12],sep='\n')
 if name=='train': print('best per output',*[sorted(scores1,key=lambda v:v[4][k])[:5] for k in range(2)],sep='\n')
 if name in ['B','C1','train']:
  X=np.c_[poly(q,1),q*q,q**3,q**5]; print('separable coeff',reg(X,Y))
