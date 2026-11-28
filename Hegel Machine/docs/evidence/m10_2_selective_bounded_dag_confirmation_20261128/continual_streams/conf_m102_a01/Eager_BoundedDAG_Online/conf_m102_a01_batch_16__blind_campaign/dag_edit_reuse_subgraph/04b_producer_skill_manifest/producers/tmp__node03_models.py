import pickle, numpy as np
from scipy.optimize import minimize
R=pickle.load(open('/tmp/node03_nested.pkl','rb'))
def design(row):
 p,b,vt,y,tr=row;s=b[:,0];d=p[:,0]-s;S=b[:,1:]@vt;D=(p[:,1:]-b[:,1:])@vt; a=S[:,-1];k=S[:,0]-.2*a;f=D[:,-1];g=D[:,0]-.2*f
 return y-s,dict(i=np.ones_like(s),d=d,s=s,s2=s*s,s3=s**3,a=a,k=k,f=f,g=g,sa=s*a,sk=s*k,a2=a*a,ak=a*k,k2=k*k,sd=s*d,ad=a*d,kd=k*d,sg=s*g,sf=s*f,ag=a*g,af=a*f,absd=abs(d),absf=abs(f),absg=abs(g),ds=d/(.035+abs(d)),df=d/(1+abs(f)/.025),dg=d/(1+abs(g)/.006),dsq=d/(1+(d/.05)**2),dp=np.maximum(d,0),dn=np.minimum(d,0),sn=np.minimum(s,0),sp=np.maximum(s,0))
groups=['i','d','i d','i d s','i d sn sp','i d s s2 s3','i d s a k','i d s f g','i d f g','i d a k f g','i d s a k sa sk a2 ak k2','i d s a k f g sd ad kd sg sf ag af','i d ds df dg dsq','i d absd absf absg','i d absd s s2 s3 a k f g','i d dp dn f g','i d s s2 s3 a k f g sa sk a2 ak k2 sd ad kd sg sf ag af absd absf absg ds df dg dsq']
results=[]
for group in groups:
 keys=group.split()
 for lam in [1,10,30,80]:
  for robust in [False,True]:
   scores=[]
   for train,valid in R:
    data=[design(row) for row in train]; r=np.concatenate([x[0] for x in data]);F=np.column_stack([np.concatenate([x[1][k] for x in data]) for k in keys]);sc=np.maximum(np.sqrt(np.mean(F**2,axis=0)),1e-8);X=F/sc;w=np.linalg.solve(X.T@X+lam*np.eye(len(keys)),X.T@r)
    if robust:
     def obj(w):
      e=(X@w-r)/.025; q=np.sqrt(1+e*e); return .025**2*np.sum(q-1)+.5*lam*(w@w),.025*(X.T@(e/q))+lam*w
     w=minimize(obj,w,jac=True,method='BFGS',options={'gtol':1e-9}).x
    v,D=design(valid);scores.append(np.mean(abs(v-np.column_stack([D[k] for k in keys])@(w/sc))))
   results.append((scores,group,lam,robust,w/sc))
for key in [lambda r:np.mean(r[0][:4])+.3*max(r[0][:4]),lambda r:r[0][-1]]:
 print('RANK');
 for scores,group,lam,robust,w in sorted(results,key=key)[:12]: print(np.round(scores,5),group,lam,robust,np.round(w,5))
