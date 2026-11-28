exec(open('/tmp/fit07.py').read().split('for name,b')[0])
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
W=w[:n]; T=np.linalg.svd(tr,full_matrices=False)[2][:3]; Y=np.c_[y,tr@T.T]; d=W.shape[1]; rng=np.random.default_rng(11)
def ring(x):return x*np.roll(x,-1,axis=1)
def fact(W,q,kind,mode):
 k=1 if mode%2==0 else 4; q=q.reshape(-1,k); z=W
 if kind=='cubic':z=W*(1+q[0]*W**2+q[1]*W**4)+q[2]*W**2
 if kind=='rat':z=(W+q[1]*W**2)/(1+q[0]**2*W**2)
 if kind=='sinh':z=np.sinh(np.clip((.1+abs(q[0]))*W,-10,10))/(.1+abs(q[0]))+q[1]*W**2
 if kind=='tanh':z=np.tanh((.1+abs(q[0]))*W)/(.1+abs(q[0]))+q[1]*W**2
 if kind=='asinh':z=np.arcsinh((.1+abs(q[0]))*W)/(.1+abs(q[0]))+q[1]*W**2
 if kind=='mix':z=W+q[0]*W**2+q[1]*W**3+q[2]*W**4
 r=ring(z)
 return np.c_[np.ones(len(W)), W,r] if mode<2 else np.c_[np.ones(len(W)), W,r,z-W, W**2, W**3]
def models(a,v):
 norm=np.array([1.,.45,.8,.8]); best=[]
 for kind,dim in [('cubic',3),('mix',3),('rat',2),('tanh',2),('asinh',2),('sinh',2)]:
  for mode in range(4):
   for pen in [.005,.05]:
    def F(q,i):return fact(W[i],q,kind,mode)
    def fn(q):
     A=F(q,a); c=reg(A,Y[a],.01);return np.r_[((A@c-Y[a])*norm).ravel(),pen*q]
    k=dim*(1 if mode%2==0 else 4); q=min([least_squares(fn,rng.normal(0,f,k),max_nfev=80) for f in [.03,.5]],key=lambda q:q.cost).x
    P=F(q,v)@reg(F(q,a),Y[a],.01)
    best.append((abs(P[:,0]-y[v]).mean(),abs(P[:,1:]@T-tr[v]).mean(),kind,mode,pen,np.linalg.norm(fn(q))))
 return sorted(best)
for a,v in folds[:2]: print('factor',models(a,v)[:8],flush=True)
R=ring(W); RD=np.c_[R,R**2,pair(R,False)]; U=np.c_[W,W**2,W**3]; combos=[('w',W),('wr',np.c_[W,R]),('r',R),('rd',RD),('u',U),('wu',np.c_[U,R]),('rdu',np.c_[RD,U])]; q=np.linalg.lstsq(np.c_[np.ones(len(u)),ring(u)],p['v1'],rcond=None)[0]; V=np.c_[np.ones(n),R]@q; combos += [('wv',np.c_[W,V]),('v',V)]
Z=W/np.maximum(W.std(0),1e-5); rr=R/np.maximum(R.std(0),1e-5)
for name,b in [('y',y),('tr',tr)]:
 out=[]
 for ka,A in combos:
  A=A/np.maximum(A.std(0),1e-5); lin=1+(A@A.T)/A.shape[1];
  for zm,zz in [('z',Z),('r',rr),('zr',np.c_[Z,rr]),('v',V/np.maximum(V.std(0),1e-5))]:
   D=cdist(zz,zz,'sqeuclidean')/zz.shape[1];
   for g,r,k,wt in itertools.product([.03,.1,.3,1,3],[.0001,.002,.03,.15,.6],['rbf','matern','rq','p2','p3'],[.2,1,5]):
    if k=='rbf': K=np.exp(-g*D)
    elif k=='matern': H=np.sqrt(g*D); K=(1+H+H**2/3)*np.exp(-H)
    elif k=='rq': K=1/(1+g*D)
    elif k=='p2': K=(1+g*(zz@zz.T)/zz.shape[1])**2
    else: K=(1+g*(zz@zz.T)/zz.shape[1])**3
    K=lin+wt*K; es=[]
    for a,v in folds:
     P=K[np.ix_(v,a)]@np.linalg.solve(K[np.ix_(a,a)]+r*np.eye(len(a)),b[a]); es.append(abs(P-b[v]).mean())
    out.append((es[0],np.mean(es),max(es),ka,zm,g,r,k,wt))
 print('kernel',name,'official',sorted(out)[:12],'mean',sorted(out,key=lambda x:x[1])[:5],flush=True)
