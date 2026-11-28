import temporal as m, numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
Y=np.c_[m.y,m.T]; N=len(Y);raw,x,c,d=[getattr(m,k)[:N] for k in ('raw','x','c','d')]; H=m.bias(np.c_[c,d]); F=m.ar._features(m.s,raw,x,c,d)
rng=np.random.default_rng(541); splits=[np.arange(52),np.arange(56),np.arange(20,72)]+[np.sort(rng.choice(N,52,False)) for _ in range(3)]
def trial(tr,wanted,reg=6e-5,amp=1.,trend=H):
 te=np.setdiff1d(np.arange(N),tr); n=len(tr);K=[]
 for a in (raw,x,np.c_[c,d]):
  a=a/a[tr].std(0);q=cdist(a,a[tr],'sqeuclidean')/a.shape[1]
  K.extend(np.exp(-w*q) for w in (.06,.2,.5,1.25,3.))
 for a in F:
  a=(a-a[tr].mean(0))/np.maximum(a[tr].std(0),1e-9);K.append(a@a[tr].T/a.shape[1])
 K.append(np.eye(N)[:,tr]); K=np.array(K); prior=np.full(len(K),-6.);prior[:15]=-4;prior[2]=0;prior[-1]=-9
 if wanted is not None:K=K[wanted];prior=prior[wanted]
 h=trend[tr];q=np.linalg.qr(h,mode='complete')[0][:,h.shape[1]:]; J=np.eye(n)-q@q.T
 flat=np.array([q@(q.T@k[tr]@q)@q.T for k in K]).reshape(len(K),-1)
 yy=Y[tr].copy(); yy[:,1:]*=amp/np.sqrt(8.)
 def obj(p):
  w=np.exp(p);P=cho_solve(cho_factor((w@flat).reshape(n,n)+J),np.eye(n))-J
  AY=P@yy;v=P.diagonal();e=AY/v[:,None];den=2*e/n/v[:,None]
  G=-(P@den)@AY.T+(P*(np.sum(den*e,axis=1))[None,:])@P
  return np.mean(np.sum(e*e,1))+reg*np.mean((p-prior)**2),w*(flat@G.ravel())+2*reg*(p-prior)/len(p)
 p=minimize(obj,prior,jac=True,method='L-BFGS-B',bounds=[(-17,6)]*(len(K)-1)+[(-17,0)],options={'maxiter':210,'ftol':1e-11}).x
 k=np.einsum('i,ijk->jk',np.exp(p),K); C=cho_factor(k[tr]);a=cho_solve(C,h);beta=m.fit(h.T@a,a.T@Y[tr]); pred=trend[te]@beta+k[te]@cho_solve(C,Y[tr]-h@beta);e=abs(pred-Y[te]);return np.array([e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()])
if __name__=='__main__':
 print('group count',len(F)); junk=len(F)+15
 for ids in ([39],[44],[52],[44,52],[39,44,52],[39,44,48,52,58]):
  h=np.c_[H,*[F[i-15] for i in ids]]; es=[]
  for tr in splits:
   te=np.setdiff1d(np.arange(N),tr);e=abs(h[te]@m.fit(h[tr],Y[tr])-Y[te]);es.append([e[:,0].mean(),e[:,1:].mean()])
  print('OLS',ids,np.round(es,4))
 sets={'full':None,'compact':[0,1,2,3,4,39,44,52,junk],'medium':list(range(15))+[39,40,41,42,43,44,45,46,47,48,52,54,58,66,70,junk]}
 configs=[(k,r,a) for k in sets for r in (6e-5,6e-6) for a in (.5,1,2,4)]
 for k,r,a in configs:
  es=np.array([trial(t,sets[k],r,a) for t in splits]);print(k,r,a,'first',np.round(es[0],5),'56/16',np.round(es[1],5),'worst',np.round(es.max(0),5),'avg',np.round(es.mean(0),5),flush=True)
