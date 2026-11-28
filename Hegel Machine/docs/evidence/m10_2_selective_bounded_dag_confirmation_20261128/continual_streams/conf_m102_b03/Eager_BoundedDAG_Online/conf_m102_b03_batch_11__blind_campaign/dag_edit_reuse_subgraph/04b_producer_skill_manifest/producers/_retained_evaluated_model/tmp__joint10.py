exec(open('/tmp/panel_search.py').read().split('C={')[0])
from scipy.optimize import least_squares
from scipy.special import expit
from itertools import combinations
np.set_printoptions(precision=4,suppress=True)
U,S,V=np.linalg.svd(t[:52],full_matrices=False); Y=np.c_[y,t@V[:2].T]; scale=Y[:52].std(0); Y=Y/scale; i=np.arange(52);j=np.arange(52,len(y)); d0=4
print('ref',a,b, 'telemetry',s['telemetry_vec']); print('scalar-state corr',np.corrcoef(Y.T))
def score(P):
 r=(P-Y)*scale;return np.mean(abs(r[j,0])),np.mean(abs(r[j,1:]@V[:2])),np.mean(abs(r[i,0])),np.mean(abs(r[i,1:]@V[:2]))
def gram(w,mode):
 c,d=(f(w) for f in funcs); dp=(c*d).sum(1); cross=np.einsum('ni,nj->nij',c,d).reshape(len(c),-1)
 F=ones(np.c_[c,d, dp if mode!='cross' else cross]); T=ones(np.c_[c,d,cross if mode!='diag' else c*d]);return F,T
for w in (z,x):
 F,T=gram(w,'dot'); A=np.c_[F@ls(F[i],Y[i,0]),T@ls(T[i],Y[i,1:])];print('res corrs',np.corrcoef((A-Y).T),score(A))
for name,fn in [('affine',lambda w:w),('tanh',np.tanh),('sin',np.sin),('soft',lambda w:w/(1+abs(w))),('warp',lambda w:w/(1+w*w))]:
 for base in [z,x]:
  for mode in ['dot','cross','diag']:
   for reg in [.01,.06,.25]:
    def pred(p):
     w=base+fn(z@p[:16].reshape(4,4)+p[16:20])
     F,T=gram(w,mode); return np.c_[F@ls(F[i],Y[i,0]),T@ls(T[i],Y[i,1:])]
    def loss(p):return np.r_[(pred(p)[i]-Y[i]).ravel(),reg*p]
    fit=least_squares(loss,np.zeros(20),max_nfev=65); print('joint',name,base is x,mode,reg,score(pred(fit.x)),flush=True)
# Fit the archive's elementary nonlinearities directly in a shared spatial basis;
# scan common frequency and sharpness, using ridge to stabilize products.
R=[]
for nm,w in [('z',z),('x',x),('Bz',z@B+h)]:
 F,T=gram(w,'dot');
 for k in [.4,.84,1.27,2.15,3.5,5.]:
  q=k*w
  for gn,g in [('soft',q/(1+abs(q))),('cos',np.cos(q)),('sin',np.sin(q)),('rat',q/(1+q*q)),('tanh',np.tanh(q)),('mix',np.c_[np.sin(q),np.cos(q),q/(1+abs(q))])]:
   for order in [0,1,2]:
    E=[g];
    if order>=1:E.append(np.einsum('ni,nj->nij',g,w).reshape(len(w),-1))
    if order==2:E.append(np.einsum('ni,nj->nij',g,g).reshape(len(w),-1))
    G=np.column_stack(E); G-=G[i].mean(0);G/=np.maximum(G[i].std(0),.02)
    for ridge in [.003,.03,.15,1.]:
     def calc(H,target):
      A=np.c_[H,G]; sc=np.maximum(np.sqrt(np.mean(A[i]**2,axis=0)),.02);A=A/sc; D=np.r_[np.full(H.shape[1],ridge*.03),np.full(G.shape[1],ridge)]; return A@np.linalg.solve(A[i].T@A[i]+np.diag(D),A[i].T@target[i]);
     Q=np.c_[calc(F,Y[:,0]),calc(T,Y[:,1:])]; R.append((*score(Q),nm,k,gn,order,ridge))
print('grid y',sorted(R)[:10]);print('grid t',sorted(R,key=lambda a:a[1])[:10]);print('grid joint',sorted(R,key=lambda a:max(a[:2]))[:10])
