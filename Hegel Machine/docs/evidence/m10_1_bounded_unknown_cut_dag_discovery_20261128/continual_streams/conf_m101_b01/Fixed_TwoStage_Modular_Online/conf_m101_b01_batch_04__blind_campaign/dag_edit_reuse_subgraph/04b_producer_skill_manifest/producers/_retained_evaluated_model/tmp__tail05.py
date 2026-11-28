from fit_dag05 import *
from scipy.optimize import least_squares
ix=np.arange(len(z)); rng=np.random.default_rng(791)
folds=[('official',ix[52:])]+[(f'r{i}',rng.choice(ix,20,replace=False)) for i in range(8)]+[(f'{j}{e}',np.argsort(z[:,j])[:12] if e=='l' else np.argsort(z[:,j])[-12:]) for j in range(4) for e in 'lh']+[('rad',np.argsort(np.linalg.norm(z,axis=1))[-16:])]
def basis(x,kind):
 s=x*x
 q={'rat':lambda:2.5*x*(1+3.66*s)/(1+6.23*s+2.01*s*s),'atan':lambda:np.arctan(2.1*x),'asinh':lambda:np.arcsinh(3.446*x)-.376*x,'tanh':lambda:np.tanh(2.25*x)+.14*x,'rat2':lambda:2.5*x*(1+3.66*s)/(1+6.23*s+2.01*s*s)+.06*x}[kind]()
 return q,np.column_stack((np.ones(len(x)),q,*[q[:,i]*q[:,j] for i in range(4) for j in range(i,4)],s/(1+.164*s)))
def predict(tr,te,kind,reg):
 q,A=basis(z,kind); B=A[tr]; Q=np.linalg.qr(B,mode='reduced')[0]; target=y[tr]-Q@(Q.T@y[tr]); x=q[tr]
 def res(p):
  h=np.tanh(x@p[:4]+p[4]); h-=Q@(Q.T@h); c=h@target/(h@h+reg**2+1e-9)
  return np.r_[target-c*h,(reg+3e-5)*p,(reg+3e-5)*c]
 rng=np.random.default_rng(45); starts=[np.zeros(5)+.1]+[rng.normal(0,s,5) for s in (.4,.8,1.5) for _ in range(3)]
 opt=min((least_squares(res,p,bounds=(-6,6),max_nfev=110) for p in starts),key=lambda o:o.cost)
 C=np.column_stack((A,np.tanh(q@opt.x[:4]+opt.x[4]))); penalty=np.full(C.shape[1],reg);penalty[0]=0
 return C[te]@np.linalg.lstsq(np.vstack((C[tr],np.diag(penalty))),np.r_[y[tr],np.zeros(C.shape[1])],rcond=None)[0]
for kind in ('rat','atan','asinh','tanh','rat2'):
 for reg in (0,.003,.015,.04,.08):
  pred=[predict(np.setdiff1d(ix,te),te,kind,reg) for _,te in folds]; err=np.array([np.mean(abs(p-y[te])) for p,(_,te) in zip(pred,folds)]); print(kind,reg,'official/worst/mean',np.round([err[0],max(err),err.mean()],5),folds[err.argmax()][0], '2h',round(err[14],5),flush=True)
