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
from scipy.optimize import least_squares
r=lambda x,j=1:np.roll(x,-j,axis=1)
base=D(np.c_[z,b*r(b)])[:n]; tgt=np.c_[Y[:,0],Y[:,1:]@F(p)[-1]]
print('BASE COEF',fit(base,tgt)); print('BASE RESID CORR',np.corrcoef((tgt-base@fit(base,tgt)).T)); print('RAW RESID SUMMARY',np.quantile(tgt-base@fit(base,tgt),[0,.1,.25,.5,.75,.9,1],axis=0))
for name in ['z','b','c','d']:
 q=X[name][:n]; print('FAMILY',name,'ranges',q.min(0),q.max(0)); print('PAIR',np.corrcoef(np.c_[q,tgt].T)[:4,4:].round(3))
B={}; S={}
for name in ['z','b','c','d','h','p','q']:
 u=X[name][:n]; B[name]=u
 for key,fn in [('sq',lambda v:v*v),('cube',lambda v:v**3),('abs',abs),('prod',lambda v:v*r(v)),('tri',lambda v:v*r(v)*r(v,2))]: B[name+key]=fn(u)
for a,bn in itertools.combinations(['z','b','c','d','h'],2):
 for i in range(4):
  S[f'{a}{bn}{i}']=B[a]*r(B[bn],i)
for i,j in itertools.combinations_with_replacement(range(4),2):
 for aa in ['z','b','c','d']:
  S[f'{aa}{i}{j}']=(B[aa][:,i]*B[aa][:,j])[:,None]
for aa in ['z','b','c','d']:
 for bb in ['z','b','c','d']:
  for k in [.3,1,3]:
   for j in range(4): S[f'{aa}/{bb}{j}/{k}']=B[aa]/(1+k*r(B[bb],j)**2)
A=np.c_[*B.values(),*S.values()]; keys=[f'{name}:{i}' for name,g in [*B.items(),*S.items()] for i in range(g.shape[1])]; keep=np.std(A,axis=0)>1e-6; A=A[:,keep];keys=np.array(keys)[keep];
from sklearn.linear_model import ElasticNet,LassoLars
for j in range(2):
 for ir,it in slices:
  x=base[ir]; q=np.linalg.qr(x)[0]; g=A[ir]-q@(q.T@A[ir]); scale=np.maximum(np.std(g,axis=0),1e-5); gt=A[it]-base[it]@fit(x,A[ir]); e=tgt[ir,j]-q@(q.T@tgt[ir,j]); initial=base[it]@fit(x,tgt[ir,j]);
  for a in [.008,.003,.001]:
   mod=ElasticNet(alpha=a,l1_ratio=.88,max_iter=4000,tol=1e-5,fit_intercept=False).fit(g/scale,e); print('elastic',j,it[0],a,np.mean(abs(initial+gt/scale@mod.coef_-tgt[it,j])),list(zip(keys[np.argsort(abs(mod.coef_))[-12:]],np.sort(abs(mod.coef_))[-12:].round(4))))
# Profile compact nonlinear wave response in a measured coordinate, with the base projected out.
for j in range(2):
 ir,it=slices[0]; x=base[ir]; q=np.linalg.qr(x)[0]; e=tgt[ir,j]-q@(q.T@tgt[ir,j]); wins=[]
 for name,u in X.items():
  for fun in ['sin','tanh','asinh','rational','cos','dead']:
   def H(par):
    v=u*par[0]+par[1]; f={'sin':np.sin,'tanh':np.tanh,'asinh':np.arcsinh,'rational':lambda x:x/(1+x*x),'cos':np.cos,'dead':lambda x:np.sign(x)*np.maximum(abs(x)-.5,0)**1.5}[fun];return np.c_[base,f(v)[:n]]
   def res(par):
    h=H(par)[ir];return h@fit(h,tgt[ir,j])-tgt[ir,j]
   for init in [[1,0],[2,.5],[4,0],[7,1]]:
    opt=least_squares(res,init,bounds=([.1,-4],[14,4]),max_nfev=35); h=H(opt.x); wins.append((np.sqrt(np.mean(opt.fun**2)),np.mean(abs(h[it]@fit(h[ir],tgt[ir,j])-tgt[it,j])),name,fun,opt.x))
 print('NONLINEAR',j,sorted(wins,key=lambda a:a[0])[:12])
