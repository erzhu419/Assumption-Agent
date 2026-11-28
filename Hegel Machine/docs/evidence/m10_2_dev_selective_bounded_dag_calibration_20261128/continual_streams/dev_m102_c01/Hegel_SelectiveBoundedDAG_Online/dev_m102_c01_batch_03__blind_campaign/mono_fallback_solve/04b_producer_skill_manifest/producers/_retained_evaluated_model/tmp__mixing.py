import numpy as np,pickle
from scipy.linalg import solve
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from adaptive import coords
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; T=s['train']['traj']; Z=coords(s,s['train']['x']); u=s['panel_c']['u']; v=s['panel_c']['v1']; w=s['panel_c']['v2']; n=len(y)
F=lambda z:np.c_[np.ones(len(z)),z]
def pf(h,v):return np.linalg.lstsq(h,v,rcond=None)[0]
ops={}
for j in range(2):
 for name,init,fun in [('oddrat',[2.],lambda x,p:x/(1+p[0]*x*x)),('rational',[2.,.2],lambda x,p:x/(1+p[0]*x*x+p[1]*x**4)),('root',[2.,.6],lambda x,p:x/(1+p[0]*x*x)**p[1]),('atan',[2.],lambda x,p:np.arctan(x*p[0])),('tanh',[2.],lambda x,p:np.tanh(x*p[0])),('even',[.16,.54],lambda x,p:x+p[1]*x*x/(1+p[0]*x*x)),('gauss',[1.],lambda x,p:x*np.exp(-p[0]*x*x))]:
  for lin in [0,1]:
   H=lambda x,p:np.c_[F(x),fun(x,p)] if lin else F(fun(x,p)); v0=[v,w][j]; obj=lambda p:(H(u,p)@pf(H(u,p),v0)-v0).ravel(); opt=least_squares(obj,init,bounds=(.0001,12),max_nfev=150); print(j,name,lin,np.std(opt.fun),opt.x,flush=True)
   if j==1 and name=='even' and not lin or j==0 and name in ('oddrat','root','atan','rational') and lin:
    a=opt.x.copy(); h=lambda x,p,fun=fun,lin=lin:np.c_[F(x),fun(x,p)] if lin else F(fun(x,p)); c=pf(h(u,a),v0); ops[str(j)+name]=lambda x,h=h,a=a,c=c:h(x,a)@c
rng=np.random.default_rng(821); folds=[(np.arange(52),np.arange(52,n))];
for _ in range(12):
 i=rng.permutation(n);folds.append((i[:52],i[52:]))
def calc(K,r):
 P=[]
 for i,j in folds: P.append(K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+r*np.eye(len(i)),y[i],assume_a='pos'))
 return np.array(P)
def score(p):
 E=abs(p-np.array([y[j] for _,j in folds])).mean(1);return np.mean(E),E[0],max(E)
res=[]; pool=[]
for k,z in enumerate(Z):
 dd=cdist(z,z,'sqeuclidean'); l=z@z.T
 for name,g in ops.items():
  q=g(z); D=cdist(q,q,'sqeuclidean'); ll=q@q.T
  for kind in ['sum','warp','features','augment']:
   for f in [.08,.3,1,3,10]:
    for scale in [.14,.23,.34,.47]:
     if kind=='sum': C=1+.4*l+np.exp(-scale*dd)+f*np.exp(-scale*D)
     if kind=='warp': C=1+.4*l+np.exp(-scale*(dd+f*D))
     if kind=='features': C=1+.4*l+np.exp(-scale*dd)+f*(1+ll)**2
     if kind=='augment':
      J=np.c_[z,q,np.einsum('ni,nj->nij',z,q).reshape(n,-1)];C=1+.4*l+np.exp(-scale*dd)+f*(J@J.T)
     for r in [3e-6,4e-5,.00025]:
      p=calc(C,r); res.append((score(p),(k,name,kind,f,scale,r),p))
 for scale in [.14,.2,.25,.32,.43]:
  for r in [3e-6,4e-5,.00025]:
   p=calc(1+.4*l+np.exp(-scale*dd),r);res.append((score(p),(k,'base','base',0,scale,r),p))
for elt in sorted(res,key=lambda e:e[0][0])[:12]:print('mean',elt[:2]);
for elt in sorted(res,key=lambda e:e[0][1])[:12]:print('gate',elt[:2]);
for subset in [lambda e:True,lambda e:e[1][1]=='base',lambda e:e[1][1]=='1even',lambda e:e[1][1].startswith('0')]:
 for k in [12,40,100]:
  top=sorted(filter(subset,res),key=lambda e:np.mean(abs(e[2][1:]-np.array([y[j] for _,j in folds[1:]]))))[:k]; print('avg',k,score(np.mean([e[2] for e in top],0)),top[0][1]);
