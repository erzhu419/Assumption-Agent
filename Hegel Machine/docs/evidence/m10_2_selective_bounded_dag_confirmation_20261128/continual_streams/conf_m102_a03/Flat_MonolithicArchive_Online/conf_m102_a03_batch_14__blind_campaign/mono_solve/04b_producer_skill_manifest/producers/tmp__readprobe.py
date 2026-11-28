import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; T=s['train']['traj']; S=np.load('/tmp/spaces.npz'); z,b=[S[k][:72] for k in ['z','B0']]; nt=52; np.set_printoptions(precision=3,suppress=True)
af=lambda u:np.c_[np.ones(len(u)),u]; fit=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-9)[0]; cross=lambda a,b:(a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def metrics(p,Y):return np.mean(np.abs(p[:52]-Y[:52])),np.mean(np.abs(p[52:]-Y[52:]))
D=np.c_[af(z),b,cross(z,b)]; print('y distribution',np.quantile(y,np.linspace(0,1,11)),'t',s['eval_times']); print('coeff scalar',fit(D[:52],y[:52])); print('coeff trajectory',fit(D[:52],T[:52])[:,[-1]].T)
H=np.c_[af(z),b]; p0=fit(D[:52],y[:52]); lnk={'tanh':np.tanh,'atan':np.arctan,'sin':np.sin,'soft':lambda v:v/(1+abs(v)),'sq':lambda v:v*v}
R=[]
for lower,F in [('bilinear',D),('affine',H)]:
 for name,fn in lnk.items():
  for style in ['same','add','index']:
   for k in [.3,1,2]:
    p=fit(F[:nt],y[:nt]); q=np.r_[p/k,.3] if style=='same' else (np.r_[p,p*k] if style=='add' else np.r_[p*k,1,1,0]);
    def pred(F,q):
     d=F@q[:F.shape[1]]
     if style=='same':return k*d+q[-1]*fn(d)
     if style=='index':return q[-3]*d+q[-2]*fn(d)+q[-1]
     return d+fn(F@q[F.shape[1]:])
    sol=least_squares(lambda q:pred(F[:nt],q)-y[:nt],q,max_nfev=250)
    R.append((*metrics(pred(F,sol.x),y),lower,name,style,k))
print('links',sorted(R,key=lambda x:x[1])[:20]);
# Alternative products: fitted operator coordinates, unmixed input warp, and paired coordinates.
from scipy.linalg import qr
for k in ['B0','B1','B2','B3']:
 b=S[k][:72]; L=np.c_[af(z),b]; Q=cross(z,b); D=np.c_[L,Q]; print(k,'linear',metrics(D@fit(D[:nt],y[:nt]),y), 'state',metrics(D@fit(D[:nt],T[:nt]),T));
 if k!='B0': continue
 for dst in [y,T[:,-1]]:
  active=list(range(L.shape[1])); pool=list(range(L.shape[1],D.shape[1])); print('greedy')
  for i in range(10):
   j=min(pool,key=lambda j:np.linalg.norm(D[:nt,active+[j]]@fit(D[:nt,active+[j]],dst[:nt])-dst[:nt])); active.append(j);pool.remove(j); print((j-L.shape[1])//4,(j-L.shape[1])%4,metrics(D[:,active]@fit(D[:nt,active],dst[:nt]),dst))
