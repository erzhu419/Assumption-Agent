import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from branch6 import *
from scipy.special import expit
F=lambda u,a,b:u+a*np.logaddexp(b*u,-b*u)-a*np.log(2)
pp=least_squares(residual,[.9,1.1],args=(F,)).x
r=A(F(w,*pp))@L(A(F(cu,*pp)),V)
_,_,vt=np.linalg.svd(Y[:52,1:],full_matrices=False)
T=np.c_[Y[:,0],Y[:,1:]@vt[0]]; T=(T-T[:52].mean(0))/T[:52].std(0)
base={'z':z,'w':w,'q':q,'r':r,'d':dead(z,*fit.x),'s':q+r,'f':q-r,'r0':A(F(z,*pp))@L(A(F(cu,*pp)),V),'qr':A(F(q,*pp))@L(A(F(cu,*pp)),V)}
def report(d): return score(d)
for k,b0 in [('wqr',np.c_[w,q,r]),('zqr',np.c_[z,q,r]),('zwqr',np.c_[z,w,q,r]),('qd',np.c_[q,dead(z,*fit.x)]),('qzr',np.c_[q,z,r])]:
 for alpha in [0,.0001,.001,.01,.04,.1,.3,1]:
  D=poly(b0,2); mu=D[:52,1:].mean(0); sd=D[:52,1:].std(0); D=A((D[:,1:]-mu)/np.maximum(sd,1e-8));
  p=D[52:]@np.linalg.solve(D[:52].T@D[:52]+alpha*52*np.diag([0]+[1]*(D.shape[1]-1))+1e-10*np.eye(D.shape[1]),D[:52].T@Y[:52]); e=abs(p-Y[52:]); print('ridge',k,alpha,np.round([e[:,0].mean(),e[:,1:].mean()],5))
features={k+str(i):b[:,i] for k,b in base.items() for i in range(4)}
for k,a in list(features.items()):
 for n,ff in [('sq',lambda a:a*a),('abs',abs),('cube',lambda a:a**3),('tanh',lambda a:np.tanh(1.25*a)),('clip',lambda a:np.clip(a,-.22,.22)),('hinge',lambda a:np.maximum(0,a-.22))]: features[n+'('+k+')']=ff(a)
for k,h in itertools.combinations_with_replacement(list(base)[:8],2):
 for i,j in itertools.product(range(4),repeat=2):
  if k==h and i>j:continue
  a,b=base[k][:,i],base[h][:,j]
  features[f'{k}{i}*{h}{j}']=a*b
  if (k,h) in [('q','r'),('w','r'),('w','q'),('q','q'),('w','w')]:
   for n,ff in [('sig',lambda a,b:a*expit(1.25*b)),('rat',lambda a,b:a/(1+1.25*b*b)),('ar',lambda a,b:a*abs(b)),('magnitude',lambda a,b:np.maximum(a,b)),('min',lambda a,b:np.minimum(a,b)),('plusRat',lambda a,b:(a+b)/(1+1.25*(a-b)**2)),('diffRat',lambda a,b:(a-b)/(1+1.25*(a+b)**2)),('sign',lambda a,b:a*np.sign(b)),('blend',lambda a,b:a*expit(1.25*(a-b))+b*expit(1.25*(b-a)))]: features[f'{k}{i}{n}{h}{j}']=ff(a,b)
names=np.array(list(features)); H=np.column_stack(list(features.values())); H-=H[:52].mean(0); H/=np.maximum(H[:52].std(0),1e-9)
for kn in ['', 'q','w','r','qr','wq','zq','zqr','wqr']:
 D0=A(np.c_[*[base[k] for k in kn]]) if kn else A(np.empty((len(Y),0)))
 for yy in [T,T[:,[0]],T[:,[1]]]:
  fitd=D0.copy(); used=[]
  for i in range(12):
   Q=np.linalg.qr(fitd[:52])[0]; S=H[:52]-Q@(Q.T@H[:52]); E=yy[:52]-Q@(Q.T@yy[:52]); val=np.linalg.norm(S.T@E,axis=1)/np.maximum(np.linalg.norm(S,axis=0),.025); val[used]=-1
   j=np.argmax(val); used.append(j);fitd=np.c_[fitd,H[:,j]]
   if i in [0,2,4,7,11]: print('sparse',kn,yy.shape[1],i+1,report(fitd),'terms',','.join(names[used]))
print('time',s['eval_times'])
