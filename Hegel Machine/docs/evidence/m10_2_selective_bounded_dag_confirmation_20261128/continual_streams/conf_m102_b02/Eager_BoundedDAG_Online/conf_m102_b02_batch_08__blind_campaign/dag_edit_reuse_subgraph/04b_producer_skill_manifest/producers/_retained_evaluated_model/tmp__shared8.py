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
r=lambda x,i=1:np.roll(x,-i,axis=1)
for name in ('panel_a','panel_b','panel_c'):print(name,s[name]['u'].shape)
u=s['panel_b']['u']; print('b check',np.max(abs(b)),s['telemetry_vec']);
base=D(np.c_[z,b*r(b)])[:n]; V=np.c_[Y[:,0],s['train']['traj']];
print('res sv',np.linalg.svd(V-base@fit(base,V),compute_uv=False));
for ir,it in slices[:2]:
 Q=np.linalg.qr(base[ir])[0]; E=V[ir]-Q@(Q.T@V[ir]); P,sv,R=np.linalg.svd(E,full_matrices=False); y=P[:,0]*sv[0]; gr={}
 xx={'z':z[:n],'b':b[:n],'c':c[:n],'d':d[:n]}; cols={k+str(i):x[:,i] for k,x in xx.items() for i in range(4)}
 for a in cols:
  x=cols[a];
  for bn in cols:
   v=cols[bn];
   gr[a+'/'+bn]=x/(1+.7*v*v)
   for cn in cols:
    if a<=bn<=cn:gr[a+'*'+bn+'*'+cn]=x*v*cols[cn]
   if a<=bn:
    gr[a+'*'+bn]=x*v
    for sign in [-1,1]:
     for coef in [.3,.7,1,1.5,2.5]:
      q=x+sign*coef*v
      for fi,f in enumerate([lambda q:q**2,lambda q:q**3,np.tanh,np.sin,np.cos,np.abs]):gr[f'{a},{bn},{sign*coef},{fi}']=f(q)
 g=np.column_stack(list(gr.values())); h=g[ir]-Q@(Q.T@g[ir]); proj=h.T@y/np.maximum(np.linalg.norm(h,axis=0),1e-9); keys=list(gr); print('TOP',it[0],[(keys[k],round(abs(proj[k])/np.linalg.norm(y),3)) for k in np.argsort(abs(proj))[-18:][::-1]])
 for sign in [-1,1]:
  for h1 in [b,c,d]:
   A=D(np.c_[z,b*r(b),z+sign*h1,(z+sign*h1)**2,(z+sign*h1)**3])[:n]; e=V[it]-A[it]@fit(A[ir],V[ir]); print('mix',it[0],sign,round(abs(e[:,0]).mean(),4),round(abs(e[:,1:]).mean(),4))
