exec(open('/tmp/stages.py').read().split('train=[',1)[0])
from scipy.optimize import minimize_scalar
I=[np.arange(52),np.arange(16,72),np.r_[0:27,43:72],np.r_[0:40,56:72]]; J=[np.setdiff1d(np.arange(72),i) for i in I]
q=poly(x,2); e=y-q@reg(q,y);print('quadratic residual by row'); print(np.c_[x,y,q@reg(q,y),e][np.argsort(abs(e))[-12:]])
for name,d in root.items():
 coef=reg(poly(d,2),y);H=np.zeros((4,4)); k=5
 for i in range(4):
  for j in range(i,4):H[i,j]=H[j,i]=coef[k]/(1 if i==j else 2); k+=1
 print('H',name,coef[:5].round(3),H.round(3).tolist(),np.linalg.eigvalsh(H).round(3))
# Evaluate compact interaction grammars with analytic training-only LOO selection.
scores=[]
for name,d in z.items():
 if len(name)>3:continue
 nls=[('id',d)]
 for f in (.5,1.,1.6,2.3):
  nls.extend([(f'tanh{f}',np.tanh(f*d)/f),(f'atan{f}',np.arctan(f*d)/f),(f'soft{f}',d/np.sqrt(1+(f*d)**2)),(f'sin{f}',np.sin(f*d)/f),(f'cos{f}',np.cos(f*d)),(f'rat{f}',d/(1+(f*d)**2))])
 for wn,v in nls:
  features=[('sq',poly(v,2)),('add',np.c_[af(x),v,v*v]),('hy',np.c_[q,v,v*v]),('cross',np.c_[q,v,(d[:,:,None]*v[:,None,:]).reshape(-1,16)])]
  if name in root:features.append(('sumcos',np.c_[q,v,np.cos(1.2*np.column_stack([d[:,i]+sgn*d[:,j] for i in range(4) for j in range(i+1,4) for sgn in (-1,1)]))]))
  for kind,F in features:
   F=F/np.maximum(.08,F.std(0));F[:,0]=1
   pred=[]; loo=[]
   for i,j in zip(I,J):
    a,b=F[i],F[j]; U,S,V=np.linalg.svd(a,full_matrices=False); u=U.T@y[i]; su=S*S; t=b@V.T
    best=(np.inf,None)
    for lam in (.001,.01,.07,.3,1.,3.):
     m=su/(su+lam); es=(y[i]-U@(m*u))/(1-U**2@m); score=np.mean(abs(es)); p=t@(S*u/(su+lam));
     if score<best[0]:best=(score,p)
    loo.append(best[0]);pred.append(np.mean(abs(best[1]-y[j])))
   scores.append((np.mean(loo),np.mean(pred),name,wn,kind,np.round(pred,5),np.round(loo,5)))
print('best by LOO');print(*sorted(scores,key=lambda v:v[0])[:15],sep='\n');print('best across slices');print(*sorted(scores,key=lambda v:max(v[5]))[:15],sep='\n')
