exec(open('/tmp/graph13.py').read().split('h=tr[')[0])
from sklearn.model_selection import KFold
b=bs['cosx'](x); y=tr['y']; tt=tr['traj']; cb=np.cos(2.6*x); h=tt[:,-1]; ts=s['eval_times']; print('times',ts,'telemetry',s['telemetry_vec']);
folds=[(np.arange(52),np.arange(52,72)),*KFold(4,shuffle=True,random_state=41).split(x)]
def cv(d,den=None):
 if den is not None:d=d/den[:,None]
 return np.array([np.mean(abs(d[j]@ls(d[i],y[i])-y[j])) for i,j in folds]);
B=quad(b); print('baseline',cv(B)); print('quad coefs',ls(B[:52],y[:52]));
dens={'b':b,'x':x,'cb':cb,'C':c1(x),'D':c2(x),'BC':c1(b),'BD':c2(b),'h':(aff(b)@ls(aff(b)[:52],h[:52]))[:,None]}; candidates=[]
for k,z in dens.items():
 d=np.c_[z,np.sum(z*z,axis=1),*((z[:,i]+c*z[:,j]) for i,j in itertools.combinations(range(z.shape[1]),2) for c in (-1.,1.))];
 for j,q in enumerate(d.T):
  for typ,T in [('sq',q*q),('abs',abs(q)),('rms',q if q.min()>=0 else q*q)]:
   def pred(i,j):
    def loss(p):
     a=B/(1+p*T[:,None]);r=a[i]@ls(a[i],y[i])-y[i];return np.mean(r*r)
    p=minimize_scalar(loss,bounds=(0,5),method='bounded').x;a=B/(1+p*T[:,None]);return np.mean(abs(a[j]@ls(a[i],y[i])-y[j])),p
   vals=np.array([pred(i,j) for i,j in folds]);candidates.append((np.mean(vals[1:,0]),vals[0,0],k,j,typ,vals[:,1].round(3)))
print('rational',*sorted(candidates,key=lambda r:r[0])[:25],sep='\n');
for k,z in dens.items():
 for nm,D in [('joint',np.c_[B,aff(z)[:,1:]]),('interaction',np.c_[B,(b[:,:,None]*z[:,None,:]).reshape(len(b),-1)]),('cubic',np.c_[B,*[z[:,i]*z[:,j]*z[:,l] for i,j,l in itertools.combinations_with_replacement(range(z.shape[1]),3)]]),('cumquad',np.c_[B,quad(z)[:,1:]])]:
  sc=cv(D); print(k,nm,sc[0],sc[1:].mean(),sc[1:].max());
print('traj',np.mean(abs(aff(b)[52:]@ls(aff(b)[:52],tt[:52])-tt[52:]),axis=0)); print('tcoef',ls(aff(b)[:52],tt[:52]).round(4));
