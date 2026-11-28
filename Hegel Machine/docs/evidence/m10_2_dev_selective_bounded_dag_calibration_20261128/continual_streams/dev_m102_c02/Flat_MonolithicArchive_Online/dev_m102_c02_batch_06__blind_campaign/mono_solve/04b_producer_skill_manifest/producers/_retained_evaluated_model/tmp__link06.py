exec(open('/tmp/search06.py').read().split('from scipy.spatial.distance')[0])
from scipy.linalg import solve
from sklearn.model_selection import KFold
F=dict(z=z,a=a,b=b,c=c,d=d,q=q,r=r)
S=np.arange(52); I=np.arange(52,72); n=len(z)
M=np.column_stack((af(a),z,np.einsum('ni,nj->nij',a,z).reshape(n,-1),terms(a,2)[0][:,5:],terms(z,2)[0][:,5:])); P=fit(M[S],Y[S]); print('COEF: scalar, terminal',np.c_[P[:,0],P[:,-1]].round(4).T); print('cross a,z terminal',P[9:25,-1].reshape(4,4).round(4)); print('cross b,z terminal',(np.linalg.solve(fit(af(a),b)[1:],P[9:25,-1].reshape(4,4))).round(4));print('cross a,z scalar',P[9:25,0].reshape(4,4).round(4));print('extremes',np.c_[Y[I,0],M[I]@P[:,0],Y[I,-1],M[I]@P[:,-1]].round(3));
def lin(H,y,tr,te,reg):
 G=H[:,1:];m=G[tr].mean(0);s=G[tr].std(0)+1e-10;G=(G-m)/s;mu=y[tr].mean(0);W=G[tr];p=solve(W.T@W+reg*np.eye(W.shape[1]),W.T@(y[tr]-mu),assume_a='pos');return G[te]@p+mu
folds=[(S,I),*KFold(4,shuffle=True,random_state=251).split(Y)]
y=Y[:,0]; T=Y[:,1:]; Hs={}
for nm,u in [('z',z),('q',q),('r',r)]:
 for threshold in [.28,.31,.35,.39]:
  for power0 in [1.25,1.5,1.75]:
   w=soft(z,[threshold,power0]); C0=np.einsum('ni,nj->nij',w,u).reshape(n,-1); A=terms(w,2)[0]; U=terms(u,2)[0];
   for mode,H in enumerate([np.c_[A,u,C0],np.c_[U,w,C0],np.c_[A,U[:,1:],C0],np.c_[af(w),u,C0]]):Hs[nm,threshold,power0,mode]=H
out=[]
for name,H in Hs.items():
 for reg in [.00001,.001,.02,.15]:
  for f in [('raw',0),('soft',.10),('soft',.22),('soft',.30),('sinh',.5),('sinh',1.),('cube',.22),('cube',.54)]:
   fn,k=f
   if fn=='soft': yy=y/(1-k*abs(y)); inv=lambda v:v/(1+k*abs(v))
   elif fn=='sinh': yy=np.sinh(k*y)/k; inv=lambda v:np.arcsinh(k*v)/k
   elif fn=='cube': yy=np.sign(y)*np.log1p(k*abs(y))/k; inv=lambda v:np.sign(v)*np.expm1(np.minimum(k*abs(v),5))/k
   else: yy=y;inv=lambda v:v
   e=[]
   for tr,te in folds:
    pp=inv(lin(H,yy,tr,te,reg));e.append(np.mean(abs(pp-y[te])))
   out.append((e[0],np.mean(e[1:]),max(e[1:]),name,reg,f))
print('best combined',*sorted(out,key=lambda e:e[0]+np.mean(e[1:3]))[:22],sep='\n');print('best gate',*sorted(out)[:8],sep='\n')
