import numpy as np,pickle
from scipy.linalg import solve
from scipy.optimize import minimize
import sys;sys.path.insert(0,'/holdout_task');from batch_solver import coords,_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x'];y=s['train']['y'];n=len(y);rng=np.random.default_rng(12345)
folds=[(np.arange(52),np.arange(52,n))];
for _ in range(12):
 i=rng.permutation(n);folds.append((i[:52],i[52:]))
yy=np.array([y[j] for i,j in folds]); I=np.eye(52)
def pred(K,r):return np.array([K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+r*I,y[i],assume_a='pos') for i,j in folds])
def met(p):
 e=abs(p-yy).mean(1);return tuple(np.round([e[0],e[1:].mean(),max(e)],6))
res=[]
for k,z in enumerate(coords(s,x)[1:3],1):
 print('baseline',k,met(np.array([_scalar(z[i],z[j],y[i]) for i,j in folds])))
 D=(z[:,None,:]-z[None,:,:])**2; dot=z@z.T; F=z/(1+.83*abs(z)); B=F@F.T; Dsum=D.sum(2)
 for a in [.16,.28,.44,.66,.95,1.4]:
  A=np.exp(-a*D); B1=A.sum(2); B2=sum(A[:,:,i]*A[:,:,j] for i in range(4) for j in range(i)); R=np.exp(-a*Dsum);
  for name,K in [('add',B1),('pairs',B2),('adddot',B1*(1+dot)),('pairsdot',B2*(1+dot)),('abs',(1+B)*(1+dot)),('sym',(1+dot)*R),('parity',R*np.cosh(2*a*dot)*np.exp(-a*Dsum)),('sqrbf',np.exp(-a*((z[:,None,:]**2-z[None,:,:]**2)**2).sum(2))*(1+dot))]:
   for w in [.04,.2,1.,5.]:
    for rr in [1e-5,1e-4,.0006]:
     P=pred(1+.35*dot+R+w*K,rr);res.append((met(P),(k,a,name,w,rr),P))
for key in [0,1,2]:print(key,*[(a,b) for a,b,p in sorted(res,key=lambda t:t[0][key])[:8]],sep='\n')
for m in [10,30,80]:
 p=np.mean([p for a,b,p in sorted(res,key=lambda t:t[0][1])[:m]],axis=0); print('pooled',m,met(p))
