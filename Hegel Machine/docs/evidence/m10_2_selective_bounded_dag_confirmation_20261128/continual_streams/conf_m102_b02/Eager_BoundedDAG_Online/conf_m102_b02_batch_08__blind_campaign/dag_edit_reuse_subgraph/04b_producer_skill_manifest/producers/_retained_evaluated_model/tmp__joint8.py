from cal8 import *
import pickle,itertools
from scipy.optimize import minimize_scalar
from scipy.linalg import qr
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z,b,c=parts(s); p=s['panel_c']; u,v=p['u'],p['v2']
H=lambda x,k:D(np.c_[x,np.cos(k*x)])
opt=minimize_scalar(lambda k:np.mean((H(u,k)@fit(H(u,k),v)-v)**2),bounds=(.2,4),method='bounded'); k=opt.x; w=fit(H(u,k),v); d=H(z,k)@w
print('C2',k,opt.fun,w.round(5)); tr=s['train']; n=len(tr['y']); ix=np.arange(n); t=s['eval_times']; T=tr['traj']; F=lambda p:-np.expm1(-t[:,None]*p); p=least_squares(lambda p:(F(p)@fit(F(p),T.T)-T.T).ravel(),[1.8,.42],bounds=([1,.15],[3,.85])).x; M=fit(F(p),T.T).T; Y=np.c_[tr['y'],M]; print('modes',p,'corr',np.corrcoef(Y.T).round(4),'RMS',Y.std(0),'trajectory resid',np.abs(F(p)@M.T-T.T).mean());
X={'z':z,'b':b,'c':c,'d':d,'bc':b+c,'bd':b+d,'cd':c+d,'bcd':b+c+d,'b-c':b-c,'b-d':b-d,'c-d':c-d,'.3z+b':.3*z+b,'.3z-b':.3*z-b,'.3z+c':.3*z+c,'.3z+d':.3*z+d,'z+b':z+b,'z+c':z+c,'z+d':z+d,'p':z*np.roll(z,-1,axis=1),'q':np.cos(k*z),'h':np.sign(z)*np.maximum(abs(z)-.35,0)**1.5}
slices=[(np.setdiff1d(ix,it),it) for it in (ix[52:],ix[:20],ix[20:40],ix[40:60])]
def scorepred(a,b):return np.r_[np.abs(a[:,0]-b[:,0]).mean(),np.abs((a[:,1:]-b[:,1:])@F(p).T).mean(axis=0).mean()]
def cv(A,reg=0):
    return np.array([scorepred(A[it]@np.linalg.solve(A[ir].T@A[ir]+reg*np.eye(A.shape[1]),A[ir].T@Y[ir]) if reg else A[it]@fit(A[ir],Y[ir]),Y[it]) for ir,it in slices])
r=lambda x,i=1:np.roll(x,-i,axis=1)
V=np.c_[Y[:,0],s['train']['traj']]; base=D(np.c_[z,b*r(b)])[:n]; names=[]; cols=[]; groups=[]
def add(name,x,g):
 names.append(name);cols.append(x[:n]);groups.append(g)
for name,x in [('z',z),('b',b),('c',c),('d',d)]:
 for i in range(4):add(name+str(i),x[:,i],name)
raw=dict(zip(names,cols));
for deg in (2,3):
 for ns in itertools.combinations_with_replacement(raw,deg):add('*'.join(ns),np.prod([raw[k] for k in ns],axis=0),''.join(k[0] for k in ns))
for a,x in raw.items():
 for bn,v in raw.items():
  for k in [.2,.6,1.5,4]:add(f'{a}/{bn}/{k}',x/(1+k*v*v),'r'+a[0]+bn[0])
for a,x in raw.items():
 for bn,v in raw.items():
  if a>=bn:continue
  for k in [-2,-1,-.5,.5,1,2]:
   for fun in [np.sin,np.tanh,np.cos,np.abs]:add(f'{fun.__name__}:{a},{k},{bn}',fun(x+k*v),'m'+a[0]+bn[0])
A=np.array(cols).T; names=np.array(names);groups=np.array(groups); stable={}
for ir,it in slices:
 x=base[ir]; Q=np.linalg.qr(x)[0]; inv=fit(x,A[ir]); g=A[ir]-Q@(Q.T@A[ir]); sc=np.maximum(np.linalg.norm(g,axis=0),1e-9); g/=sc; gt=(A[it]-base[it]@inv)/sc; E=V[ir]-Q@(Q.T@V[ir]); u,sv,v=np.linalg.svd(E,full_matrices=False); ey=u[:,0]*sv[0]; init=base[it]@fit(x,V[ir]);
 # Train-only forward beam using PRESS for the dominant shared residual direction.
 beam=[()]; lv=np.sum(Q*Q,axis=1); out=[]
 for step in range(7):
  hits={}
  for ks in beam:
   if ks:
    q=np.linalg.qr(g[:,ks])[0]; h=g-q@(q.T@g); e=ey-q@(q.T@ey); l=lv+np.sum(q*q,axis=1)
   else:h=g;e=ey;l=lv
   norm=np.linalg.norm(h,axis=0); h=h/np.maximum(norm,1e-9); pred=h*(h.T@e); er=np.mean(abs((e[:,None]-pred)/np.maximum(1-l[:,None]-h*h,.02)),axis=0);er[norm<1e-6]=np.inf
   for j in np.argpartition(er,14)[:14]:hits[tuple(sorted((*ks,j)))]=er[j]
  beam=sorted(hits,key=hits.get)[:14]; preds=[]
  for ks in beam[:7]:
   pred=init+gt[:,ks]@fit(g[:,ks],E);preds.append(pred); mae=abs(pred-V[it]).mean(0);out.append((hits[ks],mae[0],mae[1:].mean(),list(names[list(ks)])))
  mae=abs(np.mean(preds,axis=0)-V[it]).mean(0); print('beam',it[0],step,np.round([mae[0],mae[1:].mean()],4),out[-7],flush=True)
 # Tensor-polynomial kernel with explicit group scaling, keeping known base unpenalized.
 acts=[]
 for pattern in ['b','d','bd','bcd','zbd','zbcd','zd','bc','zc','zbc']:
  for deg in (2,3,4,5):
   for scl in (.3,.7,1.5,3):
    for zw in (.2,1,3):
     U=np.column_stack([raw[name]*scl*(zw if name[0]=='z' else 1) for name in raw if name[0] in pattern]); dot=U@U.T
     for kind,K in [('poly',(1+dot)**deg),('rbf',np.exp(-.5*np.maximum(0,np.diag(dot)[:,None]+np.diag(dot)-2*dot)))]:
      L=K[np.ix_(ir,ir)]; P=np.eye(len(ir))-Q@Q.T; M=P@L@P; Kt=(K[np.ix_(it,ir)]-base[it]@fit(x,L))@P
      w,Ue=np.linalg.eigh(M); w=np.maximum(w,0); D0=Ue.T@E; Lt=Kt@Ue
      for a in [1e-6,1e-4,.003,.04,.4]:
       rat=a/(w+a); err=Ue@(D0*rat[:,None]); press=np.mean(abs(err/np.maximum((Ue**2@rat-lv)[:,None],.001)),axis=0); mae=abs(init+Lt@(D0/(w+a)[:,None])-V[it]).mean(0); spec=(pattern,deg,scl,zw,kind,a); acts.append((press[0]+press[1:].mean(),mae[0],mae[1:].mean(),spec)); stable.setdefault(spec,[]).append(mae)
 print('kernel-train',it[0],sorted(acts)[:3]);print('kernel-oracle',it[0],sorted(acts,key=lambda e:e[1]+e[2])[:3]);
print('kernel-stable',sorted((max(e[0]+e[1:].mean() for e in es),[(round(e[0],4),round(e[1:].mean(),4)) for e in es],sp) for sp,es in stable.items())[:8]);
