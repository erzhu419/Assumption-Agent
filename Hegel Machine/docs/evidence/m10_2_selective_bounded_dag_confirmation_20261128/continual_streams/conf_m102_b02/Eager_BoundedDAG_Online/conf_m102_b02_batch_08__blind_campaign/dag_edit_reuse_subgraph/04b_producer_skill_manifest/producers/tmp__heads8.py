from cal8 import *
import pickle,itertools
from scipy.optimize import minimize_scalar
from scipy.linalg import qr
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z,b,c=parts(s); p=s['panel_c']; u,v=p['u'],p['v2']
H=lambda x,k:D(np.c_[x,np.cos(k*x)])
opt=minimize_scalar(lambda k:np.mean((H(u,k)@fit(H(u,k),v)-v)**2),bounds=(.2,4),method='bounded'); k=opt.x; w=fit(H(u,k),v); d=H(z,k)@w
print('C2',k,opt.fun,w.round(5)); tr=s['train']; n=len(tr['y']); ix=np.arange(n); t=s['eval_times']; T=tr['traj']; F=lambda p:-np.expm1(-t[:,None]*p); p=least_squares(lambda p:(F(p)@fit(F(p),T.T)-T.T).ravel(),[1.8,.42],bounds=([1,.15],[3,.85])).x; M=fit(F(p),T.T).T; Y=np.c_[tr['y'],M]; print('modes',p,'corr',np.corrcoef(Y.T).round(4),'RMS',Y.std(0),'trajectory resid',np.abs(F(p)@M.T-T.T).mean());
X={'z':z,'b':b,'c':c,'d':d,'bc':b+c,'bd':b+d,'cd':c+d,'bcd':b+c+d,'b-c':b-c,'b-d':b-d,'c-d':c-d,'.3z+b':.3*z+b,'.3z-b':.3*z-b,'.3z+c':.3*z+c,'.3z+d':.3*z+d,'z+b':z+b,'z+c':z+c,'z+d':z+d,'p':z*np.roll(z,-1,axis=1),'q':np.cos(k*z),'h':np.sign(z)*np.maximum(abs(z)-.35,0)**1.5}
slices=[(np.setdiff1d(ix,it),it) for it in (ix[52:],ix[:20],ix[20:40],ix[40:60])]
def scorepred(a,b):return np.r_[np.abs(a[:,0]-b[:,0]).mean(),np.abs((a[:,1:]-b[:,1:])@F(p).T).mean(axis=0).mean()]
def cv(A,reg=0):
    return np.array([scorepred(A[it]@np.linalg.solve(A[ir].T@A[ir]+reg*np.eye(A.shape[1]),A[ir].T@Y[ir]) if reg else A[it]@fit(A[ir],Y[ir]),Y[it]) for ir,it in slices])
G={}
for nm,x in X.items():
 x=x[:n]; G[nm]=x
 fun={'sq':lambda x:x*x,'cube':lambda x:x**3,'prod':lambda x:x*np.roll(x,-1,axis=1),'sumprod':lambda x:x*np.roll(x,-1,axis=1)*np.roll(x,-2,axis=1)}
 for a in [.5,1,1.5,2,3,4,6]:
  fun.update({f'{f}{a}':lambda x,f=f,a=a:f(a*x) for f in [np.sin,np.cos,np.tanh,np.arcsinh]})
 for name,f in fun.items(): G[nm+'_'+name]=f(x)
for a,b1 in itertools.combinations(['z','b','c','d','bcd'],2):
 for j in range(4):
  G[f'{a}*{b1}_{j}']=X[a][:n]*np.roll(X[b1][:n],j,axis=1)
base=D(z[:n]); out=[]
for nm,g in G.items():
 A=np.c_[base,g]; e=cv(A);out.append((e,nm))
for j in range(2):
 print('best linear groups',j,[(name,e[:,j].round(4)) for e,name in sorted(out,key=lambda en:en[0][:,j].max())[:16]])
 for j2 in [j]:
  chosen=[]; A=np.ones((n,1)); tgt=Y[:,0:1] if j==0 else Y[:,1:]@F(p).T
  for step in range(8):
   trr=ix[:52]; Q=np.linalg.svd(A[trr],full_matrices=False)[0][:,:np.linalg.matrix_rank(A[trr])]; res=tgt[trr]-Q@(Q.T@tgt[trr]);
   vals=[]
   for nm,g in G.items():
    if nm in chosen:continue
    h=g[trr]-Q@(Q.T@g[trr]); q,sv,_=np.linalg.svd(h,full_matrices=False); q=q[:,sv>1e-7]; vals.append(((q.T@res).ravel()@(q.T@res).ravel(),nm))
   chosen.append(max(vals)[1]); A=np.c_[A,G[chosen[-1]]]; print('forward',j,chosen[-1],np.sqrt(np.mean((A[trr]@fit(A[trr],tgt[trr])-tgt[trr])**2)),cv(A,1e-6)[:,j].round(4))
