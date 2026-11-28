exec(open('/tmp/chains06.py').read().split("trans={'B'")[0])
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as comb
n=len(y); print('n',n); rng=np.random.default_rng(6); splits=[(np.arange(52),np.arange(52,n)),(np.arange(20,n),np.arange(20))];
T=tr[:,-1]; Y=np.c_[y,tr]; V=np.c_[y,T];
def score(f,alpha,i,j):
 z=np.maximum(np.sqrt(np.mean(f[i]**2,axis=0)),1e-8); a=f/z; coef=np.linalg.solve(a[i].T@a[i]+alpha*np.eye(a.shape[1]),a[i].T@Y[i]); d=abs(a[j]@coef-Y[j]);return np.array([d[:,0].mean(),d[:,1:].mean()]);
Q=design(B,2); A=design(B,1); qa=design(q,1); D=C2(B); C=C1(B); H=q*np.exp(z['p'][2]*np.logaddexp(0,z['p'][1]*np.log(np.maximum(z['p'][0]*abs(q),1e-300)))); F=H-q
print('corr res',np.corrcoef((V-Q@ls(Q,V)).T)); print('coef B',ls(Q,V).round(3));
bases={'Bquad':Q,'BqProd':np.c_[Q,*[B*q[:,i:i+1] for i in range(4)]],'Bq2':np.c_[Q,design(q,2)[:,1:]],'qH':design(np.c_[q,H],2),'qF':design(np.c_[q,F],2),'qBC':np.c_[Q,design(q,2)[:,1:],*[(q-.25*C)*np.sin(q[:,i:i+1]-.25*C[:,i:i+1]) for i in range(4)]],'BD':np.c_[Q,D],'c2sep':np.c_[Q,np.tanh(1.38*B**2)],'C1':np.c_[Q,C],'Btilde':design(B1(q),2)}
for nm,L in [('q',q),('B',B),('qBC',q-.25*C),('qD',q-.25*C2(q))]:
 for phi in (np.sin,np.tanh):
  for sc in (.5,1.,2.):
   G=phi(sc*L); bases[(nm,phi.__name__,sc)]=np.c_[Q,G,*[L[:,i:i+1]*G for i in range(4)]]
def addgate(base,g,p,i,j,alpha):
 ff=lambda pp:np.c_[base,g*np.tanh(qa@pp)[:,None]]
 def residual(pp):
  a=ff(pp)[i]; z=np.maximum(np.sqrt(np.mean(a*a,axis=0)),1e-8);a=a/z; w=np.linalg.solve(a.T@a+alpha*np.eye(a.shape[1]),a.T@V[i]);return np.r_[(a@w-V[i]).ravel(),np.sqrt(alpha)*w.ravel()]
 starts=[np.r_[0,k*np.eye(4)[l]] for l in range(4) for k in (2.,7.,14.)]; o=min((least_squares(residual,p0,bounds=(-24,24),max_nfev=65) for p0 in starts),key=lambda o:o.cost); return ff(o.x),o.x,np.sqrt(np.mean(o.fun**2))
out=[]
for name,f in bases.items():
 for al in (.0001,.002,.02,.2,1.):out.append((np.max([score(f,al,i,j) for i,j in splits],axis=0),name,al))
for k in (0,1): print('LINEAR',k,sorted(out,key=lambda e:e[0][k])[:10],flush=True)
for name in ('Bquad','BqProd','Bq2','qF',('qBC','sin',1.0),('q','tanh',1.0),'Btilde'):
 for gname,g in [('const',A[:,:1]),('q',qa),('B',A),('qB',np.c_[qa,B])]:
  for al in (.002,.08):
   res=[]
   for i,j in splits:
    f,p,e=addgate(bases[name],g,None,i,j,al);res.append((score(f,al,i,j).round(5).tolist(),p.round(2).tolist(),round(e,5)))
   print('FIT',repr(name),gname,al,res,flush=True)
