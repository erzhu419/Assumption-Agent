exec(open('/tmp/search02.py').read().split('G={};')[0])
from scipy.linalg import qr
base=np.c_[np.ones(len(z)),z,cross(z),z*z]
train=np.arange(52); test=np.arange(52,72)
read=np.linalg.lstsq(np.c_[base,Y[:,1:]][train],y[train],rcond=None)[0]; Yo=Y[:,1:].copy(); Yo[:,1]=(Yo[:,1]+(y-np.c_[base,Yo[:,0]]@read[:-1])/read[-1])/2
views={n:np.column_stack([X[n+str(i)] for i in range(4)]) for n in ('z','b','i','pz','qz','pb','qb','pi','qi','mz','rz')}
def report(pred):
 yp=base[test]@read[:-2]+pred@read[-2:]; return np.round([np.mean(abs(yp-y[test])),np.mean(abs(pred@h.T-traj[test])),*np.mean(abs(pred-Yo[test]),axis=0)],5)
def fit(F,reg):
 A0=F[train]; norm=np.linalg.norm(A0,axis=0); W=np.linalg.lstsq(np.r_[A0,np.diag(norm*np.sqrt(reg))],np.r_[Yo[train],np.zeros((F.shape[1],2))],rcond=1e-9)[0];return F[test]@W
results=[]
for n,m in combinations(views,2):
 U,V=views[n],views[m]; f=(U[:,:,None]*V[:,None,:]).reshape(-1,16)
 for tag,extra in [('plain',f),('quartic',np.c_[f,U*U,V*V]),('cubic',np.c_[f,(U[:,:,None]**2*V[:,None,:]).reshape(-1,16),(U[:,:,None]*V[:,None,:]**2).reshape(-1,16)])]:
  F=np.c_[base,U,V,extra]
  for reg in [.000001,.0001,.003,.03]:
   p=fit(F,reg); results.append((list(report(p)),n,m,tag,reg))
print('linear interactions',*sorted(results)[:12],sep='\n')
for n,m in [('pz','qz'),('mz','qz'),('pz','z'),('pb','qb'),('pi','qi')]:
 U,V=views[n],views[m]; F=np.c_[base,U,V]; Q,R,P=qr(F[train],pivoting=True,mode='economic'); d=np.count_nonzero(abs(R.diagonal())>1e-8); Q=Q[:,:d];
 def res(f):return f[train]-Q@(Q.T@f[train]);
 RR=res(Yo); J=res((U[:,:,None]*V[:,None,:]).reshape(-1,16)); W=np.linalg.lstsq(J,RR,rcond=1e-9)[0];
 for j in (0,1): print('interaction sv',n,m,j,np.round(np.linalg.svd(W[:,j].reshape(4,4),compute_uv=False),4))
 for j in (0,1):
  fits=[]
  for mode in ('prod','rat','tanh','gate','cos'):
   def evalf(p):
    A=U@p[:4]; B=V@p[4:8]+p[8];
    if mode=='prod':return A*B
    if mode=='rat':return A/(1+B*B)
    if mode=='tanh':return np.tanh(A+B)*p[9]
    if mode=='gate':return A*np.tanh(B)
    return np.cos(A+B)*p[9]
   rng=np.random.default_rng(557)
   def rr(p):return np.r_[res(evalf(p))-RR[:,j],.0003*p]
   best=min((least_squares(rr,rng.normal(0,.5,10),max_nfev=180) for _ in range(6)),key=lambda x:np.linalg.norm(x.fun)); feat=evalf(best.x); coef=np.linalg.lstsq(F[train],Yo[train,j]-feat[train],rcond=1e-9)[0]; pred=F@coef+feat;fits.append((mode,round(np.mean(abs(pred[train]-Yo[train,j])),5),round(np.mean(abs(pred[test]-Yo[test,j])),5)))
  print('rank1',n,m,j,fits)
