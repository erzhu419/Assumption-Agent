from maps import *
from refine import mapB
from odd import fit as oddfit
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from itertools import combinations
tr=s['train']; n=len(tr['x']); z=inv(tr['x'],calfit(s)); b=s['panel_b'];u,v=b['u'],b['v']; Y=np.c_[tr['y'],tr['traj']];

def splineB(u,v,k,lam=.001):
 def des(x):return np.c_[bias(x),x**3,np.sign(x).reshape(-1,4,1).__mul__(np.maximum(abs(x)[...,None]-np.asarray(k),0)**3).reshape(len(x),-1)]
 F=des(u); d=np.maximum(F.std(0),.003);d[0]=1; F=F/d; C=solve(F.T@F+lam*np.diag([0]+[1]*(F.shape[1]-1)),F.T@v,assume_a='pos'); return lambda x:des(x)/d@C

def gen():
 yield 'rat',mapB(u,v,'rat','wide')[0];yield 'tanhall',fitB_all();yield 'atan',oddfit(u,v,'atan','poly')[-1];
 for l in [.00003,.001,.03]:yield 'spline'+str(l),splineB(u,v,[.25,.55,.85],l)

def fitB_all():
 from refine import fitB
 return fitB(u,v,'tanh',3.0,'all')

def kernels(w,z):
 q=w/.5; t=z/.65; dot=q@q.T/4; d=cdist(q,q,'sqeuclidean')/4; dd=cdist(t,t,'sqeuclidean')/4; quad=dot**2
 yield 'lin',1+dot;yield 'quad',1+dot+quad
 for g in [.3,1,3]:
  add=np.exp(-g*(q[:,None,:]-q[None,:,:])**2).mean(2);
  for name,r in [('full',np.exp(-g*d)),('raw',np.exp(-g*dd)),('add',add),('mix',(np.exp(-g*d)+np.exp(-g*dd)+add)/3)]:
   for h in [.03,.3,2]: yield f'{name}/{g}/{h}',1+dot+h*r
  for h in [.1,1]:yield f'rq/{g}/{h}',1+dot+h/(1+g*d)
 for h in [.1,1]:yield f'cubic/{h}',1+dot+quad+h*dot**3

if __name__=='__main__':
 ss=[np.arange(n)]+[np.random.default_rng(k).permutation(n) for k in range(10)];rs=[]
 for tag,f in gen():
  w=f(z); print('map',tag,'rmse',np.mean((f(u)-v)**2)**.5,'range',w.min(),w.max())
  for k,K in kernels(w,z):
   for l in [.0003,.003,.03,.15]:
    out=[]
    for sp in ss:
     i,j=sp[:56],sp[56:];p=K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+l*np.eye(len(i)),Y[i],assume_a='pos'); e=np.abs(p-Y[j]);out.append([e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()])
    e=np.array(out);rs.append((tag,k,l,e.mean(0),e.max(0),e[0]))
 for j in range(3):
  print('Best target',j)
  for r in sorted(rs,key=lambda r:.55*r[3][j]+.45*r[4][j])[:14]:print(r[:3],*[x.round(5) for x in r[3:]])
