from cal8 import *
import pickle,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z,b,c=parts(s); print('coord ranges',z.min(0),z.max(0))
p=s['panel_c']; u,y=p['u'],p['v2']; r=lambda u,i=1:np.roll(u,-i,axis=1)
F={'u':lambda u:u,'prod':lambda u:u*r(u),'prod2':lambda u:u*r(u,2),'sq':lambda u:u**2,'cube':lambda u:u**3,'abs':lambda u:abs(u),'diff2':lambda u:(u-r(u))**2,'sum2':lambda u:(u+r(u))**2,'triple':lambda u:u*r(u)*r(u,2),'u2v':lambda u:u*u*r(u),'uv2':lambda u:u*r(u)**2,'u3v':lambda u:u**3*r(u),'uv3':lambda u:u*r(u)**3,'uvsq':lambda u:(u*r(u))**2,'sin':lambda u:np.sin(u),'cos':lambda u:np.cos(u),'maximum':lambda u:np.maximum(u,r(u)),'minimum':lambda u:np.minimum(u,r(u))}
G={'u':lambda u:u,'v':lambda u:r(u),'v2':lambda u:r(u,2),'v3':lambda u:r(u,3),'u+v':lambda u:u+r(u),'u-v':lambda u:u-r(u),'prod':lambda u:u*r(u),'norm':lambda u:np.sqrt((u*u+r(u)**2)),'all':lambda u:np.sqrt((u*u).sum(1,keepdims=True))}
def metric(h):
    h=D(h); Q=np.linalg.svd(h,full_matrices=False)[0][:,:np.linalg.matrix_rank(h)]; res=y-Q@(Q.T@y); cv=res/(1-(Q*Q).sum(1))[:,None]; return np.mean(abs(cv))
out=[]
for keys in itertools.combinations(F,2):
    out.append((metric(np.c_[F[keys[0]](u),F[keys[1]](u)]),keys))
for n,f in F.items():
    for k,g in G.items():
        for sign in [1,-1]:
            def ff(a):return f(u)*(1+a*g(u)**2)**sign
            from scipy.optimize import minimize_scalar
            opt=minimize_scalar(lambda a:metric(ff(a)),bounds=(0,8),method='bounded'); out.append((opt.fun,(n,k,sign,opt.x)))
print('PANEL C2 models',sorted(out,key=lambda x:x[0])[:20]); print('C2 sample',y[:4]); print('C1 sample',c[:4]);
tr=s['train']; n=len(tr['y']); T=tr['traj']; yy=np.c_[tr['y'],T]; ix=np.arange(n); print('ranges y/T',np.std(yy,axis=0));
Ds={'z':z,'b':b,'c':c,'zb':np.c_[z,b],'zbc':np.c_[z,b,c],'bc':np.c_[b,c],'zc':np.c_[z,c]}
from scipy.spatial.distance import cdist
out=[]
for name,X in Ds.items():
 for transform in [False,True]:
  for scal in [.35,.7,1.,1.5]:
   U=X*scal; U=np.arcsinh(U) if transform else U; U=U[:n]; dot=U@U.T/U.shape[1]; dist=cdist(U,U,'sqeuclidean')/U.shape[1];
   for deg in [1,2,3,4]:
    for decay in [0,.1,.5,2]:
     K=(1+dot)**deg+(np.exp(-decay*dist) if decay else 0)
     for alpha in [1e-5,.001,.02]:
      er=[]
      for it in (ix[52:], ix[:20], ix[20:40],ix[40:60]):
       ir=np.setdiff1d(ix,it); pred=K[np.ix_(it,ir)]@np.linalg.solve(K[np.ix_(ir,ir)]+alpha*np.eye(len(ir)),yy[ir]); er.append(np.mean(abs(pred-yy[it]),0))
      er=np.array(er);out.append((er[:,0],er[:,1:].mean(1),(name,transform,scal,deg,decay,alpha)))
for j in [0,1]:
 for key in [lambda x:x[j].max(),lambda x:x[j][0]]:
  print('best head',j,[(a.round(4),b.round(4),c) for a,b,c in sorted(out,key=key)[:5]])
