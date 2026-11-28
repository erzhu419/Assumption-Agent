exec(open('/tmp/probe.py').read().split('ks=')[0])
from scipy.optimize import minimize_scalar
funs={'sin':np.sin,'cos':np.cos,'sin_sq':lambda x:np.sin(x*x),'cos_sq':lambda x:np.cos(x*x),'tanh':np.tanh,'atan':np.arctan,'asinh':np.arcsinh,'root2':lambda x:np.sqrt(1+x*x),'root3':lambda x:np.cbrt(1+x*x),'inv3':lambda x:np.sinh(np.arcsinh(x)/3),'exp':lambda x:np.exp(-x*x),'rat':lambda x:1/(1+x*x),'odd_rat':lambda x:x/(1+x*x),'signed_cubic':lambda x:x**3,'positive':lambda x:np.maximum(x,0),'log':lambda x:np.log1p(x*x),'sinc':lambda x:np.sinc(x/np.pi)}
def fit(x,y,fn,base,idx):
 def design(g):return np.c_[base,fn(g*x)]
 def fitat(g):
  z=design(g); q=np.linalg.lstsq(z[idx],y[idx],rcond=1e-9)[0]; return z@q-y,q
 def loss(g):return np.mean(fitat(g)[0][idx]**2)
 gg=np.linspace(.1,9,70); k=np.argmin([loss(g) for g in gg]); g=minimize_scalar(loss,bounds=(gg[max(0,k-1)],gg[min(len(gg)-1,k+1)]),method='bounded').x
 r,q=fitat(g); return np.array([abs(r[idx]).mean(),abs(r[~idx]).mean(),g]),q
for name,x,y in [('B',b['u'],b['v']),('C1',c['u'],c['v1']),('C2',c['u'],c['v2']),('Y',U[:72],Y),('T',U[:72],T)]:
 idx=np.arange(len(x)) < (18 if len(x)==24 else 52); allr=[]
 for t in ('raw','sumdiff','sq'):
  z={'raw':x,'sumdiff':np.c_[*[x[:,i]+sgn*x[:,j] for i,j in itertools.combinations(range(4),2) for sgn in [-1,1]]],'sq':x**2}[t]
  for key,fn in funs.items():
   for lin in (0,1):
    r,q=fit(z,y,fn,np.c_[np.ones(len(x)),x] if lin else np.ones((len(x),1)),idx); allr.append((r,(t,key,lin),q))
 print('\n',name)
 for r,k,q in sorted(allr,key=lambda a:a[0][1])[:8]: print(k,np.round(r,6),np.round(q,3) if name=='B' else '')
 print('Y-order samples' if name=='Y' else '',np.round(np.c_[x,y][:12],3) if name=='Y' else '')
