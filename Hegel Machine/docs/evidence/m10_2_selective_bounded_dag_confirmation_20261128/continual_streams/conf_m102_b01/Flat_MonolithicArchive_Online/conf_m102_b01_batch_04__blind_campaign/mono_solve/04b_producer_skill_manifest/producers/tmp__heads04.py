exec(open('/tmp/stress04.py').read().split("print('scalar full coeff'")[0])
from collections import defaultdict
from scipy.optimize import minimize_scalar
z=z[:72]; A=A[:72]; D=D[:72]; pair=D[:,[7,11]]; L=A[:,1:]; norm=np.sqrt(np.mean((D-D.mean(0))**2,axis=0)); norm[0]=1
rng=np.random.default_rng(677); splits=[np.arange(72)]+[rng.permutation(72) for _ in range(30)]; splits+=[np.argsort(np.linalg.norm(L,axis=1))]; scores=defaultdict(list)
print('y range',y.min(), y.max()); print('test latent bounds',m._latent(s,s['test']['x']).min(0),m._latent(s,s['test']['x']).max(0)); print('train latent bounds',z.min(0),z.max(0))
def ridge(i,j,k):
 d=D/norm; p=np.diag([0]*5+[.18,.18,.01,.18,.18,.18,.01,.18,.18,.18])*k
 return d[j]@m._ls(np.r_[d[i],np.sqrt(p)],np.r_[y[i],np.zeros(15)])
for split in splits:
 i,j=split[:52],split[52:]; out={}
 d=np.c_[A,pair]; out['pair']=d[j]@m._ls(d[i],y[i]); out['full']=D[j]@m._ls(D[i],y[i]);
 for k in [.1,1,5,20]: out['ridge'+str(k)]=ridge(i,j,k)
 for kind in ['tanh','sin','asinh','rational','softsign','atan','identity']:
  f={'tanh':np.tanh,'sin':np.sin,'asinh':np.arcsinh,'rational':lambda q:q/np.sqrt(1+q*q),'softsign':lambda q:q/(1+abs(q)),'atan':np.arctan,'identity':lambda q:q}[kind]
  for form in ['add','inside']:
   def model(p,k):
    amp=np.exp(p[-1]);
    if form=='add':return amp*f(A[k]@p[:5]/amp)+pair[k]@p[5:7]
    return amp*f(d[k]@p[:7]/amp)
   p=np.r_[m._ls(d[i],y[i]),0.]; fit=least_squares(lambda p:model(p,i)-y[i],p,bounds=([-np.inf]*7+[-3],[np.inf]*7+[5]),max_nfev=150);out[kind+form]=model(fit.x,j)
   if split is splits[0]: print(kind,form,'train RMSE',np.sqrt(np.mean(fit.fun**2)), 'params',np.round(fit.x,4))
 for k,v in out.items():
  e=abs(v-y[j]);scores[k].append([e.mean(),e.max()])
for k,v in sorted(scores.items(),key=lambda item: np.mean(item[1][:-1],axis=0)[0]):print(k,'primary',v[0],'mean',np.mean(v[:-1],0),'worst',np.max(v[:-1],0),'outer tail',v[-1])
