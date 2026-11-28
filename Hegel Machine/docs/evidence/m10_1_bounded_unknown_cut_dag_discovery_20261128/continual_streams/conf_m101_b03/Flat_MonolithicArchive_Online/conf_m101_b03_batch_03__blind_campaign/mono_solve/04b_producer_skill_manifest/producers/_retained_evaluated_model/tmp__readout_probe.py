import pickle,numpy as np
from itertools import combinations
from scipy.optimize import least_squares,minimize_scalar
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['eval_times']; X=s['train']['x']; T=s['train']['traj']; y=s['train']['y']; np.set_printoptions(precision=5,suppress=True)
def fit(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def inv(panel,x):
 u,v=panel['u'],panel['v']; r=np.roll(u,-1,axis=1); a=np.stack([np.ones_like(u),u,r,u*r,u**3],axis=-1); keep=np.max(abs(v),axis=1)<3
 c=np.stack([fit(a[keep,i],v[keep,i]) for i in range(4)]); print('cal max',np.max(abs(np.sum(a[keep]*c,axis=-1)-v[keep])))
 z=(x-c[:,0])/c[:,1]; ii=np.arange(4)
 for _ in range(16):
  r=np.roll(z,-1,axis=1); j=np.zeros((len(z),4,4)); j[:,ii,ii]=c[:,1]+c[:,3]*r+3*c[:,4]*z*z; j[:,ii,(ii+1)%4]=c[:,2]+c[:,3]*z
  z-=np.linalg.solve(j,(c[:,0]+c[:,1]*z+c[:,2]*r+c[:,3]*z*r+c[:,4]*z**3-x)[...,None])[...,0]
 return z
z=inv(s['panel_a'],X); b=s['panel_b']; B=lambda x:aff(x)@fit(aff(b['u']),b['v']); zb=B(z)
c=s['panel_c']; u=c['u']; ops={}
for name,fun,style in [('tanh',np.tanh,1),('sin',np.sin,1),('sin0',np.sin,0),('sin+cos',np.sin,2),('sin+cos+lin',np.sin,3),('cos',np.cos,0)]:
 def d(x,f):return aff(np.c_[x,fun(f*x)] if style==1 else np.c_[np.sin(f*x),np.cos(f*x)] if style==2 else np.c_[x,np.sin(f*x),np.cos(f*x)] if style==3 else fun(f*x))
 v=c['v2' if name=='cos' else 'v1']
 def loss(f):
  a=d(u,f);return np.linalg.norm(a@fit(a,v)-v)
 g=np.linspace(.25,8,180); i=np.argmin(list(map(loss,g))); f=minimize_scalar(loss,bounds=(g[max(0,i-1)],g[min(len(g)-1,i+1)]),method='bounded').x
 print(name,f,loss(f)); p=fit(d(u,f),v)
 if name in ['tanh','sin+cos','cos']:print(np.round(p,3))
 ops[name]=(d(z,f)@p,d(zb,f)@p)
def e(q):return -np.expm1(-t[:,None]*q)
def rates(ix):
 R=T[ix].T
 return least_squares(lambda q:(e(q)@fit(e(q),R)-R).ravel(),[.38,1.9,5.3],bounds=([.05,.8,3],[.8,3,10])).x
splits=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72))]+list(KFold(4,shuffle=True,random_state=92).split(X)); spl=[]
for tr,te in splits:
 E=e(rates(tr)); spl.append((tr,te,E,np.c_[y[tr],fit(E,T[tr].T).T]))
print('rates',rates(splits[0][0]),'amp std',spl[0][3].std(0))
def quad(a):
 i,j=np.triu_indices(a.shape[1]);return np.c_[a,a[:,i]*a[:,j]]
def score(D,alpha=None):
 scores=[]
 for tr,te,E,Y in spl:
  if alpha is None:P=aff(D[te])@fit(aff(D[tr]),Y)
  else:
   mu=D[tr].mean(0); sigma=np.maximum(D[tr].std(0),1e-10); A=(D-mu)/sigma; ym=Y.mean(0); ys=Y.std(0); P=np.column_stack([ElasticNet(alpha=alpha,l1_ratio=.87,max_iter=20000,tol=2e-5).fit(A[tr],v).predict(A[te]) for v in ((Y-ym)/ys).T])*ys+ym
  scores.append([np.mean(abs(P[:,0]-y[te])),np.mean(abs(P[:,1:]@E.T-T[te]))]);
 return np.round([scores[0],scores[1],np.mean(scores[2:],0),np.max(scores[2:],0)],4)
from sklearn.linear_model import Lasso
from scipy.optimize import minimize
V=c['v1']; U=c['u'];
for tag,init,fn in [('soft', [1.],lambda a,q:a/(1+q[0]*abs(a))),('softpow',[1.,1.1],lambda a,q:a/(1+q[0]*abs(a)**q[1])),('saturate',[1.,1.],lambda a,q:a/(1+q[0]*abs(a))**q[1]),('rat',[1.],lambda a,q:a/(1+q[0]*a*a)),('atan',[2.],lambda a,q:np.arctan(q[0]*a)),('asinh',[2.],lambda a,q:np.arcsinh(q[0]*a)),('tanh_shift',[2.2,0.],lambda a,q:np.tanh(q[0]*a+q[1])),('softlin',[1.,1.1],lambda a,q:np.c_[a,a/(1+q[0]*abs(a)**q[1])]),('softshift',[1.,0.],lambda a,q:(a+q[1])/(1+q[0]*abs(a+q[1]))),('signsqrt',[1.,1.],lambda a,q:a/(1+q[0]*abs(a)**q[1])**.5)]:
 def res(q):
  D=aff(fn(U,q));return (D@fit(D,V)-V).ravel()
 sol=least_squares(res,init,bounds=([.02]+[-2.]*(len(init)-1),[12.]+[5.]*(len(init)-1)),max_nfev=140); D=aff(fn(U,sol.x)); fp=aff(fn(z,sol.x))@fit(D,V)
 print('PANEL',tag,sol.x,np.linalg.norm(sol.fun),flush=True); ops[tag]=(fp,None)
g=ops['cos'][0]; ii,jj=np.triu_indices(12); labels=['x'+str(i) for i in range(4)]+['f'+str(i) for i in range(4)]+['g'+str(i) for i in range(4)]; labels+= [labels[i]+'*'+labels[j] for i,j in zip(ii,jj)]
for tag in ['soft','softpow','softlin','tanh']:
 F=quad(np.c_[z,ops[tag][0],g]); print('REG',tag,flush=True)
 for alpha in [.000015,.00005,.00015,.0005]:
  errors=[]; train_res=[]; coefs=[]
  for tr,te,E,Y in spl:
   mu=F[tr].mean(0); sd=F[tr].std(0); D=(F-mu)/sd; A=D[tr]; W=(Y-Y.mean(0))/Y.std(0)
   C=np.stack([ElasticNet(l1_ratio=.94,alpha=alpha,max_iter=60000,tol=3e-7).fit(A,w).coef_ for w in W.T],axis=1)
   P=(D[te]@C)*Y.std(0)+Y.mean(0)
   errors.append([np.mean(abs(P[:,0]-y[te])),np.mean(abs(P[:,1:]@E.T-T[te]))]); train_res.append(np.sqrt(np.mean((A@C-W)**2,axis=0))); coefs.append(C*Y.std(0)/sd[:,None])
  error=np.array(errors); print(alpha,error[0],np.max(error,axis=0),train_res[0],flush=True)
  if alpha==.00015:
   for j in range(4):
    cs=np.stack(coefs)[:,:,j]; inds=np.argsort(abs(np.median(cs,axis=0)))[-14:][::-1];print(j,' '.join('%s:%.3f/%.3f'%(labels[i],np.median(cs[:,i]),np.std(cs[:,i])) for i in inds),flush=True)
