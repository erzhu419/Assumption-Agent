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
for nm,F in {'x':X,'z':z,'B':zb,'z+cos':z+ops['cos'][0],'B+cos(B)':zb+ops['cos'][1],'tanh':ops['tanh'][0],'sin':ops['sin0'][0],'cos':ops['cos'][0]}.items():
 print('ols',nm,score(quad(F)).ravel())
for nm in ['tanh','sin','sin0','sin+cos','sin+cos+lin']:
 for idx,x in enumerate([z,zb]):
  f,g=ops[nm][idx],ops['cos'][idx]
  for variant,F in [('lin',np.c_[x,f,g]),('sum',quad(np.c_[x+f,x+g])),('cat',quad(np.c_[x,f,g])),('comp',quad(np.c_[x,f,g,x+f,x+g]))]:
   results=[(a,score(F,a)) for a in ([.00045] if variant in ['lin','sum','comp'] else [.00012,.00045,.0015])]; a,r=min(results,key=lambda ar:ar[1][2:].mean());print(nm,idx,variant,a,r.ravel())
