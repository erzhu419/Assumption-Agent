from scan import *
import sys

def dictionary(z):
 cols=[]; names=[]
 def add(v,k):
  cols.append(v); names.append(k)
 for k,X in z.items():
  for i in range(4):
   a=X[:,i]; add(a,(k,'id',i))
   for p in range(2,5): add(a**p,(k,'pow',i,p))
   for p in (.5,1,1.5,2,2.5,3,4):
    for f in (np.sin,np.cos,np.tanh,np.arctan): add(f(p*a),(k,f.__name__,i,p))
  if len(k)>2 or '1' in k or '2' in k: continue
  for i,j in combinations(range(4),2):
   a,b=X[:,i],X[:,j]; add(a*b,(k,'prod',i,j))
   add(a*a*b,(k,'sqprod',i,j)); add(b*b*a,(k,'sqprod',j,i))
   for g in (a+b,a-b,a*b):
    mode=0 if g is None else None
   for mode,g in enumerate((a+b,a-b,a*b)):
    for p in (.5,1,1.5,2,3):
     for f in (np.sin,np.cos,np.tanh,np.arctan): add(f(p*g),(k,f.__name__,i,j,mode,p))
  for i,j,l in combinations(range(4),3): add(X[:,i]*X[:,j]*X[:,l],(k,'triple',i,j,l))
  for i in range(4):
   for j in range(4):
    for p in (.3, .7, 1., 1.5, 2.): add(X[:,i]/(1+p*X[:,j]**2),(k,'ratio',i,j,p))
 return np.column_stack(cols),names

def pursue(F,Y,i,j,nsteps,base):
 A=F[i]; m=A.mean(0); norm=np.maximum(np.linalg.norm(A-m,axis=0),1e-12); A=(A-m)/norm; B=(F[j]-m)/norm
 yc=Y[i]-Y[i].mean(); P=base[i]; Q,_=np.linalg.qr(P); C=A-Q@(Q.T@A); R=yc-Q@(Q.T@yc); ids=[]; out=[]
 for t in range(nsteps):
  score=abs(C.T@R)/np.maximum(np.linalg.norm(C,axis=0),.05); score[ids]=0
  h=score.argmax(); v=C[:,h]; v=v/np.linalg.norm(v); C-=np.outer(v,v@C); R-=v*(v@R); ids.append(h)
  fit=np.column_stack((P,A[:,ids])); coef=np.linalg.lstsq(fit,yc,rcond=None)[0]
  out.append(Y[i].mean()+np.column_stack((base[j],B[:,ids]))@coef)
 return np.array(out),ids
if __name__=='__main__':
 F,names=dictionary(z); print('dictionary',F.shape)
 ids=np.arange(n); split=[(ids[:52],ids[52:]),*list(splits(n))[1:]]
 for k in ['none','x','ij','ib','b','bi']:
  for b in ([0] if k=='none' else [0,1]):
   base=np.ones((len(F),1)) if k=='none' else np.c_[np.ones(len(F)),blocks(z[k])[b]]
   metrics=[]
   for i,j in split:
    pred,sel=pursue(F,y,i,j,12,base); metrics.append(np.mean(abs(pred-y[j]),axis=1))
   M=np.asarray(metrics); print(k,b,'tail',M[0].round(4),'random',M[1:].mean(0).round(4), 'max', M.max(0).round(4)); _,sel=pursue(F,y,ids,ids,8,base); print([names[h] for h in sel])
