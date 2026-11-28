from studyfit import *
from scipy.linalg import qr
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True)
f=np.load('/tmp/features.npz');z,L,C,D,Y,H=(f[k] for k in ('z','L','C','D','Y','H')); T=Y[:,1:9];Y=Y[:,[0,9,10]]; N=52
p=s['panel_c'];u=p['u']; obj=[]
for k in [1,2]:
 fn=(lambda w,a:w/(1+a*np.abs(w))) if k==1 else (lambda w,a:w*w/(1+a*w*w));
 for j in range(3):
  def basis(U,P):
   a=P[0]; w=U+(P[1:] if len(P)>1 else 0); G=fn(w,a);return one(np.c_[U,G]) if j!=0 else one(G)
  def E(P):
   F=basis(u,P);return (F@ls(F,p['v'+str(k)])-p['v'+str(k)]).ravel()
  v=least_squares(E,[.85 if k==1 else .165]+[0.]*(4 if j==2 else 0),max_nfev=200);print('branch',k,j,v.x,np.std(v.fun))
  if j==2: obj.append(basis(z,v.x)@ls(basis(u,v.x),p['v'+str(k)]))
C,D=obj
X=[z,L,C,D]; names=['z','L','C','D']; d={}
fun={'id':lambda x:x,'sq':lambda x:x*x,'cube':lambda x:x**3,'tanh':np.tanh,'sin':np.sin,'soft':lambda x:x/(1+abs(x)), 'r':lambda x:x/(1+x*x),'r22':lambda x:x/(1+.22*x*x)}
def add(n,a):
 for i,v in enumerate(a.T):d[n+str(i)]=v
for n,x in zip(names,X):
 for fn,g in fun.items():add(fn+n,g(x))
 for i,j in itertools.combinations(range(4),2):
  d[f'{n}{i}*{j}']=x[:,i]*x[:,j]
  for fn in ['sq','tanh','r22']:
   for sign in [-1,1]:d[f'{fn}({n}{i}{sign:+}{j})']=fun[fn](x[:,i]+sign*x[:,j])
for i,j in itertools.combinations(range(4),2):
 A,B=X[i],X[j];nm=names[i]+names[j]
 for a,b in itertools.product(range(4),repeat=2):d[f'{names[i]}{a}*{names[j]}{b}']=A[:,a]*B[:,b]
 for sign in [-1,1]:
  for p in [.22,.5,1,1.25,2]:
   for fn in ['sq','tanh','r22','cube','sin']:
    add(f'{fn}({nm}{sign*p:+})',fun[fn](A+sign*p*B))
 for p in [.22,1,1.25]:
  add(f'{nm}/({p})',A/(1+p*B*B));add(f'{nm}inv/({p})',B/(1+p*A*A))
 for sh in range(4):
  P=A*np.roll(B,sh,axis=1);d[f'dot{nm}{sh}']=P.sum(1)
  for p in [.22,1,1.25]:
   for fn in ['id','tanh','r22','sq','cube']:
    add(f'{fn}({nm}prod{sh},{p})',fun[fn](p*P))
B=np.array(list(d.values())).T; keys=list(d);print('bank',B.shape)
for yid,y in enumerate(Y.T):
 for base in ['ones','CD','polyZ']:
  F={'ones':one(z[:,:0]),'CD':one(np.c_[C,D]),'polyZ':one(np.c_[z,pair(z)])}[base]
  print('GREEDY',yid,base)
  for step in range(15):
   R=B[:N]-F[:N]@ls(F[:N],B[:N]);e=y[:N]-F[:N]@ls(F[:N],y[:N]); j=np.argmax(abs(e@R)/np.maximum(np.linalg.norm(R,axis=0),1e-7));F=np.c_[F,B[:,j]]; E=abs(F@ls(F[:N],y[:N])-y); print(keys[j],round(E[:N].mean(),5),round(E[N:].mean(),5),end='; ')
  print()
# Save refined operators for structured follow-up
np.savez('/tmp/refined.npz',z=z,L=L,C=C,D=D,T=T,Y=Y,H=H,B=B,keys=keys)
