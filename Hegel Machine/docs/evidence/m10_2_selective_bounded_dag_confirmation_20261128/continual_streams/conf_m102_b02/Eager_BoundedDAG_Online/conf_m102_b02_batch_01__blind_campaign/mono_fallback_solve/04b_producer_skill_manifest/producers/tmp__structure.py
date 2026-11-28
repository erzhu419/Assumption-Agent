import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];np.set_printoptions(precision=5,suppress=True,linewidth=150)
def fit(A,B):return np.linalg.lstsq(A,B,rcond=1e-10)[0]
def add(u):return np.c_[np.ones(len(u)),u]
p=s['panel_a']; A=add(p['u']);v=p['v'];r=np.ones(len(A),bool)
for _ in range(5):
 w=fit(A[r],v[r]);r=np.linalg.norm(A@w-v,axis=1)<.001 if _ else np.linalg.norm(A@w-v,axis=1)<3*np.median(np.linalg.norm(A@w-v,axis=1))
z=np.linalg.solve(w[1:].T,(s['train']['x']-w[0]).T).T
p=s['panel_b'];u=p['u'];v=p['v'];links={'tanh':np.tanh,'sin':np.sin,'erf':erf,'rational':lambda x:x/(1+x*x),'softsign':lambda x:x/np.sqrt(1+x*x),'atan':np.arctan,'gauss':lambda x:x*np.exp(-x*x),'cubic':lambda x:x**3,'sinh':np.sinh,'asinh':np.arcsinh}
res=[]
for name,f in links.items():
 for mode in [0,1,2]:
  def F(q,u):
   a,b=q[:2]; d=q[2] if mode else 0; e=q[3] if mode==2 else 0
   return add(f(a*u+d)+b*u+e*u**3)
  def obj(q):
   M=F(q,u);return (M@fit(M,v)-v).ravel()
  os=[]
  for a in [.4,.8,1.4,2.2]:
   o=least_squares(obj,[a,-.5*a]+[0]*mode,bounds=([.1,-8]+[-2]*mode,[4,8]+[2]*mode),max_nfev=80);os.append(o)
  o=min(os,key=lambda o:np.mean(o.fun**2));M=F(o.x,u);w=fit(M,v);l=o.fun.reshape(v.shape)/(1-(M*np.linalg.pinv(M).T).sum(1))[:,None];res.append((np.sqrt(np.mean(l*l)),name,mode,o.x,F(o.x,z)@w)); print(name,mode,o.x,np.sqrt(np.mean(o.fun**2)),np.sqrt(np.mean(l*l)))
T=s['train']['traj'];t=s['eval_times']; B=np.c_[-np.expm1(-1.79505*t),-np.expm1(-.41067*t)];Y=np.c_[s['train']['y'],T@np.linalg.pinv(B).T];print('state weights',Y[:10]);print('state corr',np.corrcoef(Y.T));
np.savez('/tmp/coords.npz',z=z,Y=Y,B=B,**{f'{nm}{md}':b for _,nm,md,_,b in res});
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist
folds=[(np.arange(52),np.arange(52,72))]+list(KFold(4,shuffle=True,random_state=445).split(z)); scores=[]
for rms,nm,md,q,b in sorted(res)[:18]:
 for typ,x in [('B',b),('BZ',np.c_[b,z])]:
  for scale in [.35,.7,1.4,2.8]:
   for fn in [np.arcsinh,np.tanh,np.sin,lambda h:h/(1+h*h),lambda h:h]:
    H=fn(x*scale); X=H@H.T; L=x@x.T; Q=(H**2)@(H**2).T
    for K in [(1+X)**2+Q,(1+X+.1*L)**3+Q,(1+X+.1*L)**2+Q,1+X+Q+.1*L,np.exp(-cdist(H,H,'sqeuclidean')/2)]:
     for rg in [1e-5,.001,.02]:
      er=[]
      for i,j in folds:
       p=K[np.ix_(j,i)]@np.linalg.solve(K[np.ix_(i,i)]+rg*np.eye(len(i)),Y[i]);d=p-Y[j];er.append([abs(d[:,0]).mean(),abs(d[:,1:]@B.T).mean()]);
      e=np.array(er);scores.append((e[1:].mean(),nm,md,typ,scale,fn.__name__,e[0],e[1:].max(0)))
seen=set()
for v in sorted(scores,key=lambda v:v[0]):
 key=v[1:4]
 if key in seen:continue
 seen.add(key); print(v)
 if len(seen)>=15:break
