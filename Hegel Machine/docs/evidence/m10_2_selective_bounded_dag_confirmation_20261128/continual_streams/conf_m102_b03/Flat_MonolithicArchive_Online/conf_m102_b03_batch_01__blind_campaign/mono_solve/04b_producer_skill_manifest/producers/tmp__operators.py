import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True)
def F(x):return np.c_[np.ones(len(x)),x]
def fit(X,Y):return np.linalg.lstsq(X,Y,rcond=1e-10)[0]
a=s['panel_a']; A=[]
for u,v in zip(a['u'].T,a['v'].T):
 i,j=np.triu_indices(len(u),1); m=np.median((v[i]-v[j])/(u[i]-u[j])); r=v-m*u; keep=abs(r-np.median(r))<.001; A.append(fit(F(u[keep]),v[keep]))
a,b=np.array(A).T
B=fit(F(s['panel_b']['u']),s['panel_b']['v']); print('A',a,b,'B',B)
def Bf(u):return F(u)@B
def Binv(u):return (u-B[0])@np.linalg.inv(B[1:])
def coords(u):return {'u':u,'B':Bf(u),'Bt':Binv(u),'A':u*b+a,'Ai':(u-a)/b,'BA':Bf((u-a)/b)}
def report(P,Y,n):
 M=fit(P[:n],Y[:n]); E=abs(P[n:]@M-Y[n:]); return np.mean(E,axis=0)
def scores(P,Y,n):
 M=fit(P[:n],Y[:n]); E=abs(P[n:]@M-Y[n:]); return [E[:,0].mean(),E[:,1:].mean()]
func={'sq':lambda z:z*z,'cube':lambda z:z**3,'abs':np.abs,'cos':np.cos,'sin':np.sin,'tanh':np.tanh,'soft':lambda z:z/(1+abs(z)), 'gauss':lambda z:np.exp(-z*z),'rat':lambda z:z*z/(1+z*z),'relu':lambda z:np.maximum(z,0),'quartic':lambda z:z**4}
if __name__=='__main__':
 c=s['panel_c']; u=c['u']; V=np.c_[c['v1'],c['v2']]; res=[]
 for name,z in coords(u).items():
  for fname,fn in func.items():
   for gain in [.6,1,1.25,1.8,2.5,3.5]:
    for off in [0,.22]:
     Z=fn(gain*z+off)
     for variant,P in [('plain',F(Z)),('lin',F(np.c_[z,Z]))]:
      P0=P[:18]; e=np.abs(P[18:]@fit(P0,V[:18])-V[18:]).mean(axis=0); res.append((e[:4].mean(),e[4:].mean(),name,fname,gain,off,variant))
  D=F(np.c_[z,*[z[:,i]*z[:,j] for i in range(4) for j in range(i,4)]]); R=V-D@fit(D,V); print('C quadratic train',name,abs(R).mean(axis=0).reshape(2,4).mean(axis=1))
 for k in (0,1):print('C best',k,*sorted(res,key=lambda r:r[k])[:12],sep='\n')
 x=s['train']['x']; z=(x-a)/b; Y=np.c_[s['train']['y'],s['train']['traj']]; print('traj SVD',np.linalg.svd(Y[:,1:],compute_uv=False))
 res=[]
 for name,w in coords(z).items():
  for fname,fn in func.items():
   for gain in [.6,1,1.25,1.8,2.5,3.5]:
    for off in [0,.22]:
     v=fn(gain*w+off); cross=np.column_stack([v[:,i]*v[:,j] for i,j in combinations(range(4),2)]); mix=(w[:,:,None]*v[:,None,:]).reshape(len(w),-1)
     for variant,P in [('plain',F(v)),('lin',F(np.c_[w,v])),('qq',F(np.c_[w,v,cross,v*v])),('xq',F(np.c_[w,v,mix]))]:
      res.append((*scores(P,Y,52),name,fname,gain,off,variant))
 for k in (0,1):print('train best',k,*sorted(res,key=lambda r:r[k])[:12],sep='\n')
