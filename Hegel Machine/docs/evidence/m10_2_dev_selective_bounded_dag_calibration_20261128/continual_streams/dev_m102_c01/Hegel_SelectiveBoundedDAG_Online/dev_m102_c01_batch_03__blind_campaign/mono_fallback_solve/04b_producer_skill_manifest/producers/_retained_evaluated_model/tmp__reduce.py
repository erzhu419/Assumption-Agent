import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from scipy.special import erf
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from adaptive import coords
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; T=s['train']['traj']; Z=coords(s,s['train']['x']); u=s['panel_c']['u']; v=s['panel_c']['v1']; w=s['panel_c']['v2']; n=len(y)
F=lambda x:np.c_[np.ones(len(x)),x]
def pf(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
ops=[]
for typ,f in [('asinh',lambda x,a:np.arcsinh(a*x)),('atan',lambda x,a:np.arctan(a*x)),('tanh',lambda x,a:np.tanh(a*x)),('erf',lambda x,a:erf(a*x)),('softsign',lambda x,a:x/(1+a*abs(x))),('logabs',lambda x,a:np.sign(x)*np.log1p(a*abs(x))),('logquad',lambda x,a:x*np.log1p(a*x*x)),('root',lambda x,a:x/np.sqrt(1+a*x*x)),('even',lambda x,a:x*x/(1+a*x*x))]:
 for shift in [False,True]:
  def design(x,p,f=f,shift=shift): return np.c_[F(x),f(x+(p[4:] if shift else 0),p[:4])]
  target=w if typ=='even' else v
  def loss(p):
   M=design(u,p); return (M@pf(M,target)-target).ravel()
  op=least_squares(loss,np.r_[np.ones(4),np.zeros(4) if shift else []],bounds=(np.r_[np.full(4,.005),np.full(4,-.5) if shift else []],20),max_nfev=125); p=op.x; c=pf(design(u,p),target); print('panel',typ,shift,np.std(op.fun),np.round(p,4)); ops.append((np.std(op.fun),typ,lambda x,d=design,p=p,c=c:d(x,p)@c))
a=sorted([e for e in ops if e[1]!='even'],key=lambda e:e[0])[0][2]; b=sorted([e for e in ops if e[1]=='even'],key=lambda e:e[0])[0][2]
rng=np.random.default_rng(917); folds=[(np.arange(52),np.arange(52,n))];
for _ in range(9):
 i=rng.permutation(n);folds.append((i[:52],i[52:]))
def metrics(p):
 e=np.mean(abs(p-np.array([y[j] for i,j in folds])),axis=1);return np.mean(e),e[0],max(e)
def kernel(C,r): return np.array([C[np.ix_(j,i)]@solve(C[np.ix_(i,i)]+r*np.eye(len(i)),y[i],assume_a='pos') for i,j in folds])
res=[]
for k,z in enumerate(Z):
 for order,q in [('a',a(z)),('b',b(z)),('ab',b(a(z))),('ba',a(b(z)))]:
  X=np.c_[z,q]; K=z@z.T; L=q@q.T; D=cdist(z,z,'sqeuclidean'); E=cdist(q,q,'sqeuclidean');
  for mode in range(7):
   for scale in [.14,.23,.35,.5]:
    for weight in [.12,.5,2.]:
     A=[(1+K)*(1+weight*L), (1+K+weight*L)**3, (1+K)**2*(1+weight*L), (1+K)*(1+weight*L)**2, (1+K)*np.exp(-scale*E*weight), (1+weight*L)*np.exp(-scale*D), np.exp(-scale*(D+weight*E))][mode];
     for g in [0.,.15,1.]:
      C=A+g*np.exp(-scale*D)
      for r in [1e-6,4e-5,.0004]:
       p=kernel(C,r);res.append((metrics(p),(k,order,mode,scale,weight,g,r),p))
for key in range(3):
 print('kernels',key,*[(e,m) for e,m,p in sorted(res,key=lambda a:a[0][key])[:5]],sep='\n')
# Fold-local rational denominator, with polynomial numerator and optional panel coordinates.
for k in [1,2]:
 z=Z[k];
 for qm in ['none','a','b','ab']:
  q={'none':z,'a':a(z),'b':b(z),'ab':np.c_[a(z),b(z)]}[qm];
  for deg in [2,3]:
   X=np.c_[F(z),np.array([np.prod(z[:,j],axis=1) for d in range(2,deg+1) for j in itertools.combinations_with_replacement(range(4),d)]).T];
   if qm!='none':X=np.c_[X,q,np.einsum('ni,nj->nij',z,q).reshape(n,-1)]
   for typ in ['iso','diag','full','rank1']:
    P=[]
    for i,j in folds:
     def evaluate(p,ret=False):
      if typ=='rank1':D=1+(F(z)@p)**2
      elif typ=='iso':D=1+np.exp(p[0])*np.sum(z*z,1)
      elif typ=='diag':D=1+(z*z)@np.exp(p)
      else:
       L=np.diag(np.exp(p[:4]));L[np.tril_indices(4,-1)]=p[4:];D=1+np.sum((z@L)**2,1)
      B=X/D[:,None]; err=.005 if deg==2 else .018; c=solve(B[i].T@B[i]+err**2*np.eye(B.shape[1]),B[i].T@y[i],assume_a='pos'); return B[j]@c if ret else np.r_[B[i]@c-y[i],err*c,.022*p if typ=='rank1' else .005*(p-p0)]
     p0={'iso':[-2.2],'diag':[-2.2]*4,'full':[-1.1]*4+[0]*6,'rank1':[0,.2,.2,.2,.2]}[typ]; p0=np.array(p0); p=least_squares(evaluate,p0,bounds=(-6,3),max_nfev=60).x;P.append(evaluate(p,True))
    print('rational',k,qm,deg,typ,metrics(np.array(P)),flush=True)
