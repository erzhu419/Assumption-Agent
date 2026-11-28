import temporal as m, numpy as np,itertools
from scipy.linalg import cho_factor,cho_solve
Y=np.c_[m.y,m.T]; N=len(Y); S=np.arange(N)<52
names=[];fs=[]
def add(name,z):
 if z.ndim==1:z=z[:,None]
 fs.append(z[:N]);names.extend([name+':'+str(i) for i in range(z.shape[1])])
raw,x,b,c,d=[getattr(m,k) for k in ('raw','x','b','c','d')]
D=dict(raw=raw,x=x,b=b,c=c,d=d)
for k,z in list(D.items()):
 add(k,z)
 for p in (2,3,4):add(k+'^'+str(p),np.column_stack([np.prod(z[:,j],1) for j in itertools.combinations_with_replacement(range(4),p)]))
for (k,u),(l,v) in itertools.combinations(D.items(),2):
 add(k+'*'+l,(u[:,:,None]*v[:,None,:]).reshape(len(x),-1))
 for sg in (1,-1):
  zz=(u[:,:,None]+sg*v[:,None,:]).reshape(len(x),-1);DUMMY=k+('+' if sg==1 else '-')+l
  for w in (.5,1.,1.2,1.6,2.,2.4,3.,4.8):
   add('cos'+str(w)+DUMMY,np.cos(w*zz));add('sin'+str(w)+DUMMY,np.sin(w*zz))
for k,z in list(D.items())+[('x+b',x+b),('x-b',x-b),('c+d',c+d),('c-d',c-d),('c*d',c*d)]:
 for w in (.35,.5,.75,1,1.2,1.5,1.6,1.8,2,2.4,3,4.8):
  for f,g in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('rat',lambda a:a/(1+a*a)),('absrat',lambda a:a/(1+abs(a)))]:add(f+str(w)+k,g(w*z))
for j,z in enumerate(m.ar._features(m.s,raw,x,c,d)):add('archive'+str(j),z)
F=np.column_stack(fs);F=(F-F[S].mean(0))/np.maximum(F[S].std(0),1e-9)
U,V,W=np.linalg.svd(m.T[S],full_matrices=False); tar=np.c_[m.y,m.T@W[:3].T];tar=tar-tar[S].mean(0)
for j in range(4):
 r=tar[S,j].copy(); A=F[S]; J=[];print('OMP',j)
 for t in range(16):
  ij=np.argmax(abs(A.T@r));J.append(ij);w=np.linalg.lstsq(A[:,J],tar[S,j],rcond=None)[0];r=tar[S,j]-A[:,J]@w
  if t<9 or t==15:print(t,names[ij],round(np.mean(abs(r)),5),round(np.mean(abs(F[~S][:,J]@w-tar[~S,j])),5))
for i,f in enumerate(fs[-len(m.ar._features(m.s,raw,x,c,d)):]):
 A=np.c_[np.ones(N),c[:N],d[:N],f]; p=A[~S]@np.linalg.lstsq(A[S],Y[S],rcond=1e-8)[0];e=np.abs(p-Y[~S]); print('archive group',i,round(e[:,0].mean(),4),round(e[:,1:].mean(),4))
