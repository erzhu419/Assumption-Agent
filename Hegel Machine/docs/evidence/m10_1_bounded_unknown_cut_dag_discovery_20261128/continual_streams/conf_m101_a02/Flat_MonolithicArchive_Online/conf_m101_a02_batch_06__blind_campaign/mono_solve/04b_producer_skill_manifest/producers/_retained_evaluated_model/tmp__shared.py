from operators import *
np.set_printoptions(precision=4,suppress=True)
b,c,d=[ops[k](z) for k in 'BCD']; A=np.c_[aff(z),b,c,d]; n=52
R=Y[:n]-A[:n]@fit(A[:n],Y[:n]); U,S,V=np.linalg.svd(R,full_matrices=False); print('JOINT',S,V[:4]);
for i in range(1,9):
 D=np.c_[A[:72],Y[:,i]]; print('CONTRAST',i,np.mean(abs(D[n:72]@fit(D[:n],Y[:n,0])-Y[n:,0])),fit(D[:n],Y[:n,0]));
from scipy.linalg import qr
for panel,key,func,p0 in [('panel_b','v',lambda x,p:x*(1-p[1]*np.exp(-p[0]*x*x)),[1,1.1]),('panel_c','v1',lambda x,p:np.tanh(p[0]*x),[2.15])]:
 u,v=s[panel]['u'],s[panel][key];
 def run(terms):
  def design(p):return np.c_[aff(func(u,p)),terms]
  def r(p):
   a=design(p);return (a@fit(a,v)-v).ravel()
  p=least_squares(r,p0,max_nfev=100).x; rr=r(p).reshape(v.shape); a=design(p); h=(qr(a,mode='economic')[0]**2).sum(1);return np.mean(abs(rr/(1-h[:,None]))),p
 terms=[(k,pol(u)[:,i]) for i,k in enumerate(mon) if len(k) in (2,3)]; terms += [('all squares',u*u),('all linear',u),('all cosine',np.cos(1.27*u)),('all prod',np.prod(u,1)),('all cyclic pair',u*np.roll(u,-1,axis=1))]; print(panel,sorted([(run(term),str(k)) for k,term in terms],key=lambda e:e[0][0])[:10]);
# Readout ablations: mixed quadratic terms and stabilized higher-degree terms.
from scipy.linalg import solve
from itertools import combinations
cs={'z':z,'B':b,'C':c,'D':d}
cs.update({k+l:f(cs[k]) for k in 'BCD' for l,f in ops.items()});
results=[]
for keys in [('z',),('B',)]+list(combinations(cs,2))+[('z','B','C'),('z','B','D'),('B','C','D'),('z','B','C','D')]:
 a=np.column_stack([cs[k] for k in keys]); a=(a-a.mean(0))/a.std(0); Gs=[a@a.T/a.shape[1],sum((a[:,4*j:4*(j+1)]@a[:,4*j:4*(j+1)].T/4)**2 for j in range(len(keys)))/len(keys)]; G=Gs[0]; Q=G**2; P=Gs[1];
 for power in (0,.1,1):
  for cross in (0,.1,1):
   K=1+G+P+cross*Q+power*(P*G+G**3);
   for lam in (.0001,.001,.01,.08):
    ss=[]
    for i,j in folds:
     R=abs(K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+lam*np.eye(len(i)),Y[i],assume_a='pos')-Y[j]);ss.append([R[:,0].mean(),R[:,1:].mean()])
    results.append((np.array(ss),keys,power,cross,lam))
for t in (0,1):print('BEST K',t,[(np.round(a,4).tolist(),*b) for a,*b in sorted(results,key=lambda r:r[0][:,t].max())[:8]])
