import itertools,pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=np.load('/tmp/feat15.npz'); z,w,h,d=[a[k] for k in ['z','b','bc','bd']]; Y=a['Y']; n=52
D=lambda *x:np.c_[np.ones(len(x[0])),*x]
L=lambda x,y:np.linalg.lstsq(x,y,rcond=1e-9)[0]
def run(X,y):
 c=L(X[:n],y[:n]); e=X@c-y;return np.mean(abs(e[:n])),np.mean(abs(e[n:])),np.mean(e[:n]**2)**.5
p=np.sum(z*h,1); base=D(z,h,p); blocks={'z':z,'w':w,'h':h,'d':d,'c2h':a['bcd']}; cols={'p':p,'p2':p*p,'p3':p**3}
for k,v in blocks.items():
 for j in range(4):cols[k+str(j)]=v[:,j]
for (k,v),(l,u) in itertools.combinations_with_replacement(blocks.items(),2):
 for r in range(4):cols[k+l+'dot'+str(r)]=np.sum(v*np.roll(u,r,axis=1),1)
 for i,j in itertools.product(range(4),repeat=2): cols[k+str(i)+l+str(j)]=v[:,i]*u[:,j]
for i in range(3):
 y=Y[:,i]; print('TARGET',i,'base',run(base,y),'coef',L(base[:n],y[:n]),flush=True)
 for r in range(4):
  ranking=sorted((run(np.c_[base,x],y),k) for k,x in cols.items()); ranking.sort(key=lambda r:r[0][2]); print('greedy',ranking[:7],flush=True); base=np.c_[base,cols[ranking[0][1]]]
 base=D(z,h,p)
 xc={k:v for k,v in cols.items() if k in ['p','p2'] or 'dot' in k or len(k)<=3}; out=[]
 for k,q in xc.items():
  q=q/max(.001,np.std(q))
  for name,fun in [('sin',lambda u,t:np.sin(t*u)),('cos',lambda u,t:np.cos(t*u)),('tanh',lambda u,t:np.tanh(t*u)),('rat',lambda u,t:u/(1+(t*u)**2)),('sat',lambda u,t:1/(1+(t*u)**2))]:
   for bkey,b in [('base',base),('plus',np.c_[base,z[:,1]*h[:,0]])]:
    def mat(t):return np.c_[b,fun(q,t[0])]
    def err(t):
     X=mat(t)[:n];return X@L(X,y[:n])-y[:n]
    f=min((least_squares(err,[t],bounds=(.02,10),max_nfev=35) for t in [.3,1.1,2.8,5.0]),key=lambda m:m.cost);out.append((run(mat(f.x),y),k,name,bkey,f.x))
 print('nonlinear',sorted(out,key=lambda r:r[0][2])[:12],flush=True)
 for fn in (np.sin,np.tanh,np.arcsinh,lambda x:x**2,lambda x:x/(1+x*x)):
  for b in (base,np.c_[base,z[:,1]*h[:,0]],np.c_[base,w,d]):
   q=L(b[:n],y[:n]); X=b[:n]; G=np.linalg.pinv(X); U=b[:,1:]; ut=U[:n];
   def err(par):
    v=par[-1]*fn(ut@par[:-1]);u=y[:n]-v;return X@(G@u)-u
   rng=np.random.default_rng(222); starts=[np.r_[q[1:]*c,t] for c,t in itertools.product([.3,1.,2.],[.3,1.])]+[rng.normal(size=b.shape[1])*.3 for k in range(10)]; f=min((least_squares(err,p,max_nfev=200) for p in starts),key=lambda f:f.cost); v=f.x[-1]*fn(U@f.x[:-1]); pred=b@(G@(y[:n]-v[:n]))+v; print('index',getattr(fn,'__name__','?'),b.shape[1],np.mean(abs(pred[:n]-y[:n])),np.mean(abs(pred[n:]-y[n:])),f.cost,flush=True)
print('telemetry',s['telemetry_vec'], 'summary',s['train']['summary'])
