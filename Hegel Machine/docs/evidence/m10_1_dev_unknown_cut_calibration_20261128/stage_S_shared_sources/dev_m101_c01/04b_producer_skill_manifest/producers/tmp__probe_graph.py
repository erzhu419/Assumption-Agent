import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
from itertools import combinations_with_replacement as cr
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']
def aff(x):return np.c_[np.ones(len(x)),x]
def poly(x,d):return np.c_[aff(x),*[np.prod(x[:,i],axis=1) for k in range(2,d+1) for i in cr(range(x.shape[1]),k)]] if d>1 else aff(x)
def fit(a,y):return np.linalg.lstsq(a,y,rcond=None)[0]
def loo(a,y):
 p=np.linalg.pinv(a);return (a@(p@y)-y)/np.maximum(1-np.sum(a*p.T,axis=1)[:,None],1e-8)
p=s['panel_c'];u=p['u'];v=p['v1']
fams={'asinh':lambda x,c:np.arcsinh(c*x),'atan':lambda x,c:np.arctan(c*x),'tanh':lambda x,c:np.tanh(c*x),'softabs':lambda x,c:x/(1+c*abs(x)),'erf':lambda x,c:erf(c*x),'rational':lambda x,c:x/(1+c*x*x),'rootsign':lambda x,c:x/np.sqrt(1+c*x*x),'sin':lambda x,c:np.sin(c*x),'logsign':lambda x,c:np.sign(x)*np.log1p(c*abs(x))}
models=[]
for k,f in fams.items():
 for form in ('odd','odd+lin','odd+shift','odd+cube'):
  def d(x,q):
   z=f(x+(q[1:] if form=='odd+shift' else 0),q[0]);return aff(z) if form in ('odd','odd+shift') else np.c_[aff(z),x if form=='odd+lin' else x**3]
  q=min((least_squares(lambda q:loo(d(u,q),v).ravel(),[a]+([0]*4 if form=='odd+shift' else []),bounds=([.02]+([-1]*4 if form=='odd+shift' else []),[12]+([1]*4 if form=='odd+shift' else [])),max_nfev=75).x for a in [.5,2]),key=lambda q:np.mean(loo(d(u,q),v)**2));e=np.mean(loo(d(u,q),v)**2)**.5
  models.append((e,k,form,q))
print('C1 best',sorted(models,key=lambda a:a[0])[:12]);_,k,form,q=min(models,key=lambda a:a[0]);f=fams[k]
def dc(x):
 z=f(x+(q[1:] if form=='odd+shift' else 0),q[0]);return aff(z) if form in ('odd','odd+shift') else np.c_[aff(z),x if form=='odd+lin' else x**3]
c1w=fit(dc(u),v);c2w=fit(np.c_[aff(u),np.arcsinh(.77*u)**2],p['v2']);p=s['panel_b'];bw=fit(aff(p['u']),p['v']);p=s['panel_a'];r=np.arange(len(p['u']));a=aff(p['u']);
for _ in range(4):
 aw=fit(a[r],p['v'][r]);r=np.argsort(np.linalg.norm(a@aw-p['v'],axis=1))[:-3]
ops={'A':lambda x:aff(x)@aw,'I':lambda x:(x-aw[0])@np.linalg.inv(aw[1:]),'B':lambda x:aff(x)@bw,'J':lambda x:(x-bw[0])@np.linalg.inv(bw[1:]),'C':lambda x:dc(x)@c1w,'D':lambda x:np.c_[aff(x),np.arcsinh(.77*x)**2]@c2w}
x=s['train']['x'];Y=np.c_[s['train']['y'],s['train']['traj']];xx={'x':x}
for _ in range(3):
 for key in list(xx):
  if len(key)!=_+1:continue
  for k,op in ops.items():
   if (key[-1]+k) in ('AI','IA','BJ','JB'):continue
   xx[key+k]=op(xx[key]);
I=np.arange(56);J=np.arange(56,72);T=s['eval_times'];U=np.c_[-np.expm1(-1.52*T),-np.expm1(-.22*T)];P=U@np.linalg.pinv(U)
def scores(a):
 w=fit(a[I],Y[I]);w[:,1:]=w[:,1:]@P;r=abs(a[J]@w-Y[J]);return np.array([r[:,0].mean(),r[:,1:].mean(),r[:,-1].mean()])
S=[]
for k,z in xx.items():
 for m,a in [('poly2',poly(z,2)),('poly3',poly(z,3)),('x+poly2',np.c_[aff(x),poly(z,2)[:,1:]]),('x2+odd',np.c_[poly(x,2),np.arcsinh(z),np.tanh(z)]),('x2+even',np.c_[poly(x,2),np.arcsinh(z)**2]),('x+cross',np.c_[aff(x),z,x*z,z*z])]:
  S.append((scores(a),k,m))
for n in (0,1,2):
 print('rank',n,[(k,m,e.round(5).tolist()) for e,k,m in sorted(S,key=lambda row:row[0][n])[:12]])
print('baseline',[(d,scores(poly(x,d))) for d in (1,2,3)])
