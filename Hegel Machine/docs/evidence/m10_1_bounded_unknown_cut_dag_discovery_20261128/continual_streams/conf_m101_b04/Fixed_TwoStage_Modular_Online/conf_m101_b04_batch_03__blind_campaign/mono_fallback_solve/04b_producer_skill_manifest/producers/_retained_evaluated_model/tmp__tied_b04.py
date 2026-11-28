exec(open('/tmp/operators.py').read().split('ops={}')[0])
from scipy.special import erf
from scipy.spatial.distance import cdist
from scipy.linalg import solve
p=s['panel_b']; u,v=p['u'],p['v']; rng=np.random.default_rng(41); folds=np.array_split(rng.permutation(len(u)),6)
rng=np.random.default_rng(710); ids=[(np.arange(72),52),(np.arange(72),56),*[(rng.permutation(72),56) for _ in range(8)],*[(np.argsort(abs(z[:,j])),52) for j in range(4)]]
fun={'sin':np.sin,'tanh':np.tanh,'erf':erf,'rat':lambda v:v/(1+v**2),'exp':lambda v:v*np.exp(-v**2)}
def panel(name,i):
 f=fun[name]; start={'sin':[.4,2.8],'tanh':[.74,1.4],'erf':[.75,1.4],'rat':[.9,1.],'exp':[.9,1.]}[name];
 def D(U,q):return lin(U-q[0]*f(q[1]*U))
 def err(q):
  A=D(u[i],q);return (A@fit(A,v[i])-v[i]).ravel()
 o=least_squares(err,start,bounds=([.05,.3],[2.,4.])); q=o.x; C=fit(D(u[i],q),v[i]); return lambda U:D(U,q)@C,q,np.mean(o.fun**2)**.5

def run(F,B,kernel=None,shrink=0):
 out=[]
 for ix,n in ids:
  i,j=ix[:n],ix[n:]; f=F[i]; g=F[j]; yy=Y[i]
  if kernel is not None:
   K=np.exp(-kernel*cdist(z,z,'sqeuclidean')); A=K[np.ix_(i,i)]+.13*np.eye(len(i)); fi=solve(A,f,assume_a='pos'); yi=solve(A,yy,assume_a='pos'); beta=fit(f.T@fi,f.T@yi); pred=g@beta+K[np.ix_(j,i)]@(yi-fi@beta)
  elif shrink:
   sd=np.maximum(np.std(f,axis=0),.05); f=f/sd;g=g/sd; beta=np.linalg.solve(f.T@f+shrink*np.diag(np.r_[np.zeros(5),np.ones(f.shape[1]-5)]),f.T@yy);pred=g@beta
  else:pred=g@fit(f,yy)
  E=abs(pred-Y[j]);out.append([E[:,0].mean(),E[:,1:].mean()])
 out=np.asarray(out);return np.round(np.r_[out[:2].ravel(),out[2:10].max(0),out[10:].ravel()],5)
for name in fun:
 h,q,e=panel(name,np.arange(len(u))); E=[]
 for j in folds:
  hh,*_=panel(name,np.setdiff1d(np.arange(len(u)),j));E.extend((hh(u[j])-v[j]).ravel())
 b=h(z); P=poly(b,2); C=fit(P,Y); print(name,q,'panel RMS',np.round([e,np.mean(np.square(E))**.5],6),'quad',run(P,b));print('coef y, time first, time final',C[:,[0,1,-1]].round(3).T)
 if name in ('sin','tanh','erf'):
  for lab,F in [('sparse',P[:,[0,1,2,3,4,7,11]]),('bounded sparse',np.c_[P[:,:5],np.tanh(b[:,0])*np.tanh(b[:,2]),np.tanh(b[:,1])*np.tanh(b[:,3])]),('bounded all',np.c_[P[:,:5],poly(np.tanh(b),2)[:,5:]])]:
   print(lab,run(F,b), 'kernel',run(F,b,.18))
  for l in [.1,1.,5.]:print('ridge',l,run(P,b,shrink=l))
