exec(open('/tmp/operator01.py').read().split('for base in')[0])
from scipy.spatial.distance import cdist
from scipy.linalg import solve
folds=[np.arange(52,72),*np.array_split(np.arange(len(x)),4),np.argsort(np.max(np.abs(x),axis=1))[-16:]]

def metrics(pred):
 return np.round([np.mean(np.abs(pred(np.setdiff1d(np.arange(len(x)),te),te)-y[te])) for te in folds],5)

def ls(A):
 return metrics(lambda tr,te:A[te]@np.linalg.lstsq(A[tr],y[tr],rcond=1e-9)[0])

def ridge(A,K,pen,offset):
 def predict(tr,te):
  G=K[np.ix_(tr,tr)]+pen*np.eye(len(tr)); D=A[tr]; P=solve(G,np.c_[y[tr]-offset[tr],D],assume_a='pos'); c=np.linalg.lstsq(D.T@P[:,1:],D.T@P[:,0],rcond=1e-8)[0]; return offset[te]+A[te]@c+K[np.ix_(te,tr)]@(P[:,0]-P[:,1:]@c)
 return metrics(predict)

A=np.c_[ones,b]; AB=np.c_[ones,x,b]; W=np.c_[x,b]; design=np.c_[ones,x,b]; c=np.linalg.lstsq(A,y,rcond=None)[0]; e=y-A@c
print('linear coefficients',c,'error quantiles',np.quantile(e,[0,.1,.5,.9,1])); print('raw residual correlation',np.corrcoef(np.c_[e,poly(x,2)[:,5:],poly(b,2)[:,5:]].T)[0,1:]); print('eval times',t)
q=[]
for f in (poly(x,2)[:,5:],poly(b,2)[:,5:], (x[:,:,None]*b[:,None,:]).reshape(len(x),-1)):
 D=f-design@np.linalg.lstsq(design,f,rcond=None)[0]; coef=np.linalg.lstsq(D,e,rcond=None)[0]; print('quadratic residual coefficients',coef,'res MAE',ls(np.c_[design,f])); q.append(D)
for j in range(4):
 for k in range(j):
  for d in (b[:,j]*b[:,k],x[:,j]*x[:,k]):
   if abs(np.corrcoef(e,d)[0,1])>.2: print('pair',j,k,np.corrcoef(e,d)[0,1],ls(np.c_[AB,d]))
results=[]
for base,offset,label in [(A,0*y,'b'),(AB,0*y,'xb'),(ones[:,None],b@[-1,1,1,1],'signed')]:
 for wr, G in [('x',x),('b',b),('xb',W),('signed',np.c_[b,x*.3])]:
  for mode in ('rbf','matern','quad','cubic','separable'):
   for scale in (.25,.5,1.,2.,4.):
    Z=G/np.std(G,axis=0)/np.sqrt(G.shape[1]); R=cdist(Z,Z)/scale
    if mode=='rbf': K=np.exp(-.5*R**2)
    elif mode=='matern': K=(1+R+R**2/3)*np.exp(-R)
    elif mode=='separable': K=np.exp(-.5*((Z[:,None]-Z[None,:])/scale)**2).sum(2)
    else: K=(1+Z@Z.T/scale**2)**(2 if mode=='quad' else 3)
    for r in (1e-4,.001,.01,.05,.2,1.):
     m=ridge(base,K,r,offset); results.append((max(m[:5]),m.mean(),(label,wr,mode,scale,r),m))
print('best smooth residual by validation');
for elem in sorted(results,key=lambda z:z[:2])[:18]: print(elem)
