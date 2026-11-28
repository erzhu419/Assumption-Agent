from probe import *
import sys,itertools
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
sys.path.insert(0,'/holdout_task'); import node_01 as orig
X=np.vstack((s['train']['x'],s['test']['x'])); Z,B=[p[j] for j in ('z','b')]; C,D=orig._panels(u,v1,v2,B); Cz,Dz=orig._panels(u,v1,v2,Z)
Q=np.linalg.qr(np.c_[-np.expm1(-1.56*t),-np.expm1(-.38*t),np.ones(len(t))])[0]; Y=np.c_[y,tr@Q]; splits=[np.arange(52,72),*np.array_split(np.random.default_rng(733).permutation(72),3)]

def library(x,z,b,c,d,cz,dz,n,style):
 k=orig._kernels(x,z,b,c,d,n); groups=[]
 def norm(a):return (a-a[:n].mean(0))/a[:n].std(0).clip(1.e-8)
 def add(a):
  a=norm(a);groups.append(a@a.T/a.shape[1]);return a
 P=[x,z,b,c,d,cz,dz]; F=[]
 for i,a in enumerate(P):
  a=norm(a); f=[a, np.tanh(a*1.5),np.sin(a),a/(1+1.5*a*a),a*a];F.append([norm(j) for j in f]);
  for mode in range(4):
   q=a[:,mode:mode+1];add(np.c_[q,np.sin(q),np.tanh(q),q/(1+q*q),q**3,q**5]);add(np.c_[q*q,np.cos(q),np.cos(2*q),q**4]);
 for i,(f,g) in enumerate(itertools.combinations_with_replacement(range(len(P)),2)):
  if f>=4 or g>=6:continue
  a,aa,_,_,a2=F[f]; b,bb,_,_,b2=F[g];
  for j in range(4):
   for l in range(4):
    if f==g and j>=l:continue
    A,AA,A2=a[:,j],aa[:,j],a2[:,j]; B,BB,B2=b[:,l],bb[:,l],b2[:,l]
    add(np.c_[A*B,AA*B,A*BB,AA*BB]);
    if style==2 or f in (2,3) or g==3:add(np.c_[A2*B,A*B2,A2*B2]);
 for j in range(len(P)):
  if j<4:continue
  a=norm(P[j]); K=a@a.T/4;groups.extend([(1+K)**2/4,(1+K)**3/8])
 return np.concatenate((k,np.array(groups)))

def pred(k,Y,noise,pen,method):
 n=len(Y);m=len(k); mu=Y.mean(0); sd=Y.std(0);y=(Y-mu)/sd
 a=np.concatenate((np.eye(n)[None],k[:,:n,:n])); R=np.ascontiguousarray(a.reshape(m+1,-1)); ident=np.eye(n); prior=np.log(np.r_[noise,np.full(m,1./m)]); r=y.shape[1]
 def calc(theta):
  v=np.exp(theta); cov=(v@R).reshape(n,n)+1.; cf=cho_factor(cov,check_finite=False); alpha=cho_solve(cf,y,check_finite=False); ic=cho_solve(cf,ident,check_finite=False)
  if method=='loo':
   h=ic.diagonal(); e=alpha/h[:,None]; ga=-2*ic@(e/h[:,None]); gh=-2*np.mean(e*e,1)/h; g=(ga@alpha.T/r)-(ic*gh)@ic; loss=np.sum(e*e)/r
  else:
   loss=np.log(cf[0].diagonal()).sum()+.5*np.sum(alpha*y)/r;g=.5*(ic-alpha@alpha.T/r)
  d=theta-prior;return loss+pen*(d@d),v*(R@g.ravel())+2*pen*d
 theta=minimize(calc,prior,jac=True,method='L-BFGS-B',bounds=[(-12.,1.)]+[(-15.,3.)]*m,options={'maxiter':85,'ftol':2.e-7}).x
 v=np.exp(theta);cf=cho_factor((v@R).reshape(n,n)+1.,check_finite=False); alpha=cho_solve(cf,y,check_finite=False)
 return ((1+np.einsum('k,kij->ij',v[1:],k[:,n:,:n]))@alpha)*sd+mu

def evaluate(a,j):
 r=abs(a-Y[j]);return np.r_[r[:,0].mean(),abs((a[:,1:]-Y[j,1:])@Q.T).mean(),r.mean(0)[1:]]

def main():
 best=[]
 for jj,j in enumerate(splits):
  i=np.setdiff1d(np.arange(72),j); ix=np.r_[i,j]; n=len(i); ps=[]
  K0=orig._kernels(*[v[ix] for v in (X,Z,B,C,D)],n); K1=library(*[v[ix] for v in (X,Z,B,C,D,Cz,Dz)],n,1)
  for style,k in enumerate((K0,K1)):
   for pen,met in ((.008,'ml'),(.035,'ml'),(.12,'ml'),(.003,'loo'),(.03,'loo')):
    for joint in (0,1,2):
     blocks=([[0],[1],[2],[3]],[[0,1],[2],[3]],[[0,1,2],[3]])[joint]; out=np.zeros((len(j),4))
     for cols in blocks:out[:,cols]=pred(k,Y[np.ix_(i,cols)],.004 if cols[0]<3 else .04,pen,met)
     key=(style,pen,met,joint);ps.append(out);print(jj,key,*np.round(evaluate(out,j),5),flush=True)
  np.save('/tmp/refine'+str(jj),ps)
if __name__=='__main__':main()
