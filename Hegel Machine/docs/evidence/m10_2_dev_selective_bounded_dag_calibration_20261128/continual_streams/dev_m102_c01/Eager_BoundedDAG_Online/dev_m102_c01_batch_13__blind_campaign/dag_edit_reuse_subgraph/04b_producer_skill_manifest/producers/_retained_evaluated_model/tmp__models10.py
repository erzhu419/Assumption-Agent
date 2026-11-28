from search10 import *
from scipy.optimize import minimize,nnls
from sklearn.preprocessing import PolynomialFeatures
D=pickle.load(open('/tmp/data10','rb'))
def build(p,state,x,q):
 h={}; score=p['scores'][:,0]; qs=(state@p['modes'].T)[:,0]; scale=max(score.std(),1e-6); t=score/scale; tq=qs/scale
 h['t']=(np.outer(t,t),np.outer(tq,t)); h['K']=(p['K'],p['Kq']); h['1']=(np.ones((len(t),len(t))),np.ones((len(tq),len(t))))
 for name,F,Q in [('x',x,q),('sy',p['sy'],p['qy']),('st',p['st'][:,:4],p['qt'][:,:4])]:
  mu=F.mean(0); F,Q=F-mu,Q-mu
  for order in range(1,5):
   f=PolynomialFeatures(order,include_bias=False); A=f.fit_transform(F); B=f.transform(Q); A,B=A[:, -int(__import__('math').comb(F.shape[1]+order-1,order)):],B[:, -int(__import__('math').comb(F.shape[1]+order-1,order)):]; h[name+str(order)]=(A@A.T,B@A.T)
  d=cdist(F,F,'sqeuclidean'); dq=cdist(Q,F,'sqeuclidean');
  for k in [.07,.2,.45,1.]:
   h[name+'r'+str(k)]=(np.exp(-k*d),np.exp(-k*dq));
   if name=='x':
    u=F/np.maximum(F.std(0),.2); v=Q/np.maximum(F.std(0),.2); a=(u[:,None,:]-u[None,:,:])**2; b=(v[:,None,:]-u[None,:,:])**2; h['add'+str(k)]=(np.exp(-k*a).sum(2),np.exp(-k*b).sum(2))
 return h
B=[build(p,state,x,q) for p,state,x,q,_ in D]
def predict(h,y,l,rob=0):
 a,b=h; lam=np.full(len(y),l)
 for it in range(4 if rob else 1):
  c=cho_factor(a+np.diag(lam)); z=cho_solve(c,y)
  if it==3 or not rob:break
  v=cho_solve(c,np.eye(len(y))).diagonal(); r=abs(z/np.sqrt(v)); lam=l+np.maximum((r/max(rob*1.4826*np.median(r),1e-6))**2-1,0)/v
 return b@z
rng=np.random.default_rng(33); models=[]
keys=list(B[0]);
for k in range(8800):
 d={'1':1.,'t':.01}; kind=k%5
 if kind==0:d.update(x1=1.,x2=rng.choice([.3,1.]),x3=10**rng.uniform(-2,.5),x4=10**rng.uniform(-3,0))
 if kind==1:d.update(x1=1.,x2=.3)
 if kind==2:d['K']=1.
 if kind==3:d.update(sy1=1.,sy2=.3,x1=1.,x2=.3)
 if kind==4:d.update(x1=1.,x2=.3,x3=.1,sy2=10**rng.uniform(-2,1))
 for nm in rng.choice(keys[3:],rng.integers(0,4),replace=False):d[nm]=10**rng.uniform(-2,1.6)
 l=10**rng.uniform(-5,-.4); rob=rng.choice([0,0,0,2,3]); vals=[]; preds=[]
 for bb,(p,_,_,_,truth) in zip(B,D):
  a,b=(sum(w*bb[nm][i] for nm,w in d.items()) for i in (0,1)); yy=p['y_centered']+p['y_mean']; pr=predict((a,b),yy,l,rob); vals.append(np.mean(abs(pr-truth))); preds.append(pr)
 models.append((max(vals),np.mean(vals),vals[0],d,l,int(rob),np.array(vals),preds))
for key in [lambda s:(max(s[0],s[2]*1.2),s[1]),lambda s:s[1],lambda s:s[2]]:
 print('leaders');
 for m in sorted(models,key=key)[:5]: print(m[:7]);
 for count in [5,25,100,300]:
  group=sorted(models,key=key)[:count]; print('ensemble',count,[round(np.mean(abs(np.mean([m[-1][i] for m in group],0)-D[i][-1])),5) for i in range(len(D))])
pickle.dump(models,open('/tmp/models10out','wb'))
