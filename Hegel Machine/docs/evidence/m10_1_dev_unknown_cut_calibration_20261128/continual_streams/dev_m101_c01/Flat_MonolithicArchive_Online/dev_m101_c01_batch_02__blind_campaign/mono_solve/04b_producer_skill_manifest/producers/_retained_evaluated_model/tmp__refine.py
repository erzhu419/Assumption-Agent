from operators import *
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from scipy.special import erf
np.set_printoptions(precision=5,suppress=True)
c=S['panel_c']; u=c['u']; v=c['v1'];
func={'asinh':np.arcsinh,'atan':np.arctan,'erf':erf,'tanh':np.tanh,'softsign':lambda w:w/(1+abs(w)),'root':lambda w:w/np.sqrt(1+w*w),'bend':lambda w:np.sqrt(1+w*w),'sin':np.sin}
res=[]
for name,h in func.items():
 for shift in (False,True):
  def design(w,p):return np.c_[affine(w),h(w*np.exp(p[0])+(p[1] if shift else 0))]
  def err(p):
   f=design(u,p);return (v-f@ls(f,v)).ravel()
  p=least_squares(err,np.array([.5,0.] if shift else [.5]),bounds=(-3.,3.)).x; f=design(u,p); b=ls(f,v); r=v-f@b; lev=(np.linalg.qr(f)[0]**2).sum(1); e=np.sqrt(np.mean((r/(1-lev[:,None]))**2)); print(name,shift,p,e,np.linalg.svd(b[5:],compute_uv=False)); res.append((e,h,p,b,shift))
e,h,p,b,shift=min(res,key=lambda q:q[0]); c1=lambda w:np.c_[affine(w),h(w*np.exp(p[0])+(p[1] if shift else 0))]@b
f2=lambda w:w+np.logaddexp(1.1*w,-1.1*w)/1.1
q=ls(affine(f2(u)),c['v2']); c2=lambda w:affine(f2(w))@q
x=S['train']['x']; y=S['train']['y']; z=S['train']['traj']; t=S['eval_times']; rng=np.random.default_rng(451); splits=[(np.arange(56),np.arange(56,72))]+[(p[:56],p[56:]) for p in [rng.permutation(72) for _ in range(8)]]; modes_fold=[modes(t,z[tr]) for tr,te in splits]; Z=[(m@ls(m,z[tr].T)).T for m,(tr,te) in zip(modes_fold,splits)]
cs=coords(x,S); kernels={}; base=np.c_[affine(x),powers(x,2)]; P=base@base.T
for name,w in cs.items():
 if name not in ('raw','A','Ai','BA','BAi','BiA','BiAi','B','Bi'):continue
 for label,W in [('w',w),('c1',c1(w)),('c2',c2(w)),('c12',np.c_[c1(w),c2(w)]),('wc12',np.c_[w,c1(w),c2(w)])]:
  W=W/np.sqrt(W.shape[1]/4); D=cdist(W,W,'sqeuclidean');
  for length in (.7,1.1,1.6,2.3,3.4):
   R=np.sqrt(5*D)/length
   for tag,K in [('rbf',np.exp(-D/(2*length**2))),('m52',(1+R+R*R/3)*np.exp(-R))]:
    kernels[name,label,length,tag]=K
errs=[]; E=np.eye(len(x))
for key,K in kernels.items():
 for w in (.1,1.):
  for noise in (1e-5,1e-4,1e-3,1e-2):
   G=P+w*K+noise*E; er=[]
   for (tr,te),zt in zip(splits,Z):
    pred=G[np.ix_(te,tr)]@solve(G[np.ix_(tr,tr)],np.c_[y[tr],zt],assume_a='pos'); a=abs(pred-np.c_[y[te],z[te]]); er.append(np.r_[a[:,0].mean(),a[:,1:].mean(),a[:,-1].mean()]);
   errs.append((key+(w,noise),np.array(er)))
for j in range(3):
 print('target',j)
 for k,a in sorted(errs,key=lambda p:np.mean(p[1][:,j])+.35*np.max(p[1][:,j]))[:12]: print(k,a[0],a.mean(0),a.max(0))
 for k,a in sorted((p for p in errs if p[0][:2]==('raw','w')),key=lambda p:np.mean(p[1][:,j])+.35*np.max(p[1][:,j]))[:2]: print('raw',k,a[0],a.mean(0),a.max(0))
