exec(open('/tmp/probe.py').read().split('rows=[]')[0])
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
splits=[np.arange(72)]+[np.random.default_rng(i+177).permutation(72) for i in range(16)]
I,J=np.triu_indices(4)
def gram(f):return f@f.T
z=Q['A-']; zs=[z,Q['A-B'],Q['A-Bi']]; d2=[cdist(v,v,'sqeuclidean') for v in zs]; lin=gram(z); quad=gram(z[:,I]*z[:,J]); one=np.ones_like(lin); parts0=[one,lin,quad]
for i,z in enumerate(zs):
 z=z/np.sqrt(np.mean(z*z,axis=0));
 for k in (.4,.8,1.5,2.5):
  for name,w in [('soft',z/(1+k*abs(z))),('tanh',np.tanh(k*z)),('root',z/np.sqrt(1+k*z*z)),('rat',z/(1+k*z*z))]:
   w=w/np.sqrt(np.mean(w*w,axis=0)); l=gram(w); q=gram(w[:,I]*w[:,J]); m=gram((z[:,:,None]*w[:,None,:]).reshape(len(z),-1));
   parts0.append((i,k,name,l,q,m))
base=parts0[:3]; alt=parts0[3:]; g=[np.exp(-d2[0]/k) for k in (1,2,4,8,16,32)];
def make(parts,noise_bounds=(-11,-4)):
 P=np.array(parts); npar=len(parts); result=[]; predictions=[]
 for ix in splits:
  a,b=ix[:52],ix[52:]; pa=P[:,a][:,:,a]; pb=P[:,b][:,:,a]; E=np.eye(len(a)); rr=[]; pp=[]
  for yy in [y,tt]:
   v=yy[a];
   def fun(h):
    w=np.exp(h); C=cho_factor(np.einsum('i,ijk->jk',w[:-1],pa)+w[-1]*E); al=cho_solve(C,v); inv=cho_solve(C,E); G=(inv-np.outer(al,al))/2; return v@al/2+np.log(np.diag(C[0])).sum(),w*np.r_[np.einsum('ij,kij->k',G,pa),np.trace(G)]
   init=np.r_[np.full(npar,-3.),-8.]; r=minimize(fun,init,jac=True,method='L-BFGS-B',bounds=[(-15,5)]*npar+[noise_bounds],options={'maxiter':155}); w=np.exp(r.x); pred=np.einsum('i,ijk->jk',w[:-1],pb)@cho_solve(cho_factor(np.einsum('i,ijk->jk',w[:-1],pa)+w[-1]*E),v); rr.append(np.mean(abs(pred-yy[b]))); pp.append(pred)
  result.append(rr); predictions.append(pp)
 a=np.array(result); return a[0],a.mean(0),a.max(0),np.array(predictions)
rows=[]
for name,P in [('base',base+g),('poly',base+[lin**k for k in range(3,8)]),('ANOVA',[one]+[sum((zs[0][:,j,None]*zs[0][None,:,j])**p for j in range(4)) for p in range(1,7)]+[quad]+g)]+[(str((i,k,n)),base+g+[l,q,m]) for i,k,n,l,q,m in alt]:
 a,b,c,pred=make(P); rows.append((name,a,b,c,pred))
for j in (0,1):
 print('OUT',j); print(*[r[:4] for r in sorted(rows,key=lambda r:r[2][j])[:14]],sep='\n');
for label,rs in [('all',rows),('warped',rows[3:]),('raw',rows[3:19]),('forward',rows[19:35]),('inverse',rows[35:]),('best',sorted(rows,key=lambda r:r[2][0])[:10])]:
 pred=np.mean([r[-1] for r in rs],axis=0); truth=np.array([Y[ix[52:]].T for ix in splits]); a=np.mean(abs(pred-truth),axis=2); print('ENSEMBLE',label,a[0],a.mean(0),a.max(0))
