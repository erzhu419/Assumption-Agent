from panels import *
from scipy.optimize import least_squares
from sklearn.model_selection import KFold
rng=np.random.default_rng(25)
splits=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72)),*KFold(4,shuffle=True,random_state=13).split(X)]

def nl(z,kind):
 if kind=='sin': return np.sin(z),np.cos(z)
 if kind=='tanh':
  a=np.tanh(z);return a,1-a*a
 if kind=='soft': return z/(1+abs(z)),(1+abs(z))**-2
 if kind=='rat': return z/(1+z*z),(1-z*z)/(1+z*z)**2
 if kind=='even': return 1/(1+z*z),-2*z/(1+z*z)**2
 if kind=='quart': return z*z/(1+abs(z)**2),2*z/(1+z*z)**2

def fit(d,i,j,ks,degree,reg,seed=32,nstart=10):
 r=np.random.default_rng(seed); q=PolynomialFeatures(degree).fit_transform(d); xx=aug(d); F=q[i]; A=xx[i]; y=Y[i]; m=F.shape[1]; k=len(ks); n=len(i)
 penal=np.r_[np.full(m,reg),np.full(k,reg),np.full(k*5,reg*.08)]**.5; penal[0]=0
 def calc(p,xx= A,qq=F,jac=False):
  ww=p[m+k:].reshape(k,5); z=xx@ww.T; h,dh=[np.stack(a,1) for a in zip(*(nl(z[:,a],kind) for a,kind in enumerate(ks)))]; c=p[m:m+k];
  if jac:return np.r_[np.c_[qq,h,(dh[:,:,None]*c[None,:,None]*xx[:,None,:]).reshape(len(xx),-1)],np.diag(penal)]
  return qq@p[:m]+h@c
 def err(p):return np.r_[calc(p)-y,penal*p]
 opt=[]
 for rep in range(nstart):
  w=r.normal(0,.75,(k,5)); w[:,0]*=.75
  if rep<4:
   w[:,1:]=np.roll(np.eye(4),rep,axis=0)[:k]*r.uniform(.6,2); w[:,0]*=.5
  H=np.stack([nl(A@w[a],kind)[0] for a,kind in enumerate(ks)],1); p=np.r_[ls(np.c_[F,H],y),w.ravel()];
  opt.append(least_squares(err,p,jac=lambda p:calc(p,jac=True),max_nfev=150,ftol=1e-7))
 opt.sort(key=lambda a:a.cost); ps=np.array([calc(a.x,xx[j],q[j]) for a in opt[:3]]); return ps,opt[0]

if __name__=='__main__':
 print('B',B,'A',ab,flush=True)
 print('panels A/B residual',rms(aug(b['u'])@B-b['v']),flush=True)
 configs=[]
 for nm,d in [('X',X),('Ai',(X-ab[:,0])/ab[:,1])]:
  for ks in [('sin',),('sin','sin'),('sin','sin','sin'),('tanh','tanh'),('soft','soft'),('rat','rat'),('sin','even'),('sin','tanh'),('sin','rat'),('sin','sin','even'),('tanh','even')]:
   for deg,reg in [(2,.0003),(1,.0003),(2,.005)]:
    ps,o=fit(d,*splits[0],ks,deg,reg,nstart=12); er=np.mean(abs(ps-Y[splits[0][1]]),axis=1); ee=np.mean(abs(ps.mean(0)-Y[splits[0][1]])); configs.append((min(er[0],ee),nm,d,ks,deg,reg)); print('ridge',nm,ks,deg,reg,round(o.cost,5),np.round(er,4),round(ee,4),flush=True)
 for _,nm,d,ks,deg,reg in sorted(configs,key=lambda a:a[0])[:12]:
  err=[]
  for i,j in splits:
   p,o=fit(d,i,j,ks,deg,reg,nstart=18); err.append([abs(p[0]-Y[j]).mean(),abs(p.mean(0)-Y[j]).mean()]);
  print('CV',nm,ks,deg,reg,np.array(err).round(4).tolist(), 'param',o.x.round(3).tolist(),flush=True)
