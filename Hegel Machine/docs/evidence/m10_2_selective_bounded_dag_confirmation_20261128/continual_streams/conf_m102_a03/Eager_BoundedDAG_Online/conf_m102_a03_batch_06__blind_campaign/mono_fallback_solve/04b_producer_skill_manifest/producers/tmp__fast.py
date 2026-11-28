import pickle,sys,numpy as np
from scipy.optimize import minimize_scalar,least_squares
from scipy.linalg import eigh
sys.path.insert(0,'/holdout_task');from batch_solver import *
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; h=s['train']; z=cal(s['panel_a'],h['x']); x=h['x']; b,p,q=ops(s,z); y=h['y']; t=h['traj']; n=52
u=s['panel_b']['u'];v=s['panel_b']['v'];
forms={'sin':lambda z,k:z-np.sin(k*z)/k,'tanh':lambda z,k:z-np.tanh(k*z)/k,'atan':lambda z,k:z-np.arctan(k*z)/k,'rat':lambda z,k:z**3/(1+k*z*z),'sqrt':lambda z,k:z-z/np.sqrt(1+k*z*z),'sinh':lambda z,k:z-np.arcsinh(k*z)/k,'erf':lambda z,k:z-np.sqrt(np.pi)/2*__import__('scipy').special.erf(k*z)/k,'cube':lambda z,k:z**3,'power':lambda z,k:z*abs(z)**k}
def loo(a,v):
 Q=np.linalg.qr(a)[0];return np.mean(((v-Q@(Q.T@v))/(1-(Q*Q).sum(1))[:,None])**2)
for name,f in forms.items():
 opt=minimize_scalar(lambda k:loo(aff(f(u,k)),v),bounds=(.002,4),method='bounded'); print('B',name,opt.x,opt.fun**.5,flush=True)

def std(a):
 a=a-a[:n].mean(0);return a/np.maximum(a[:n].std(0),1e-9)
def kernel(a):
 a=std(a);return a@a.T/a.shape[1]

def run(z,b,p,q,label):
 ks={}; lin=kernel(z); quad=kernel(poly(z)[:,1:]); ix,jx=np.triu_indices(4,1); products=z[:,ix]*z[:,jx]; cross=kernel(products); out=[]; allx=np.c_[z,b,p,q];
 for coord,xx in [('z',z),('b',b),('zp',np.c_[z,p]),('zb',np.c_[z,b]),('zq',np.c_[z,q]),('all',allx)]:
  ks[coord+'quad']=kernel(poly(xx)[:,1:]); ks[coord+'cross']=kernel(pol if (pol:=xx[:,np.triu_indices(xx.shape[1],1)[0]]*xx[:,np.triu_indices(xx.shape[1],1)[1]]).shape[1] else xx)
  for k in [.25,.5,1,2,3,4,6,9]:
   ks[coord+'sep'+str(k)]=kernel(np.c_[xx,xx*xx,xx**3,np.tanh(k*xx),np.sqrt(.05+xx*xx),np.sin(k*xx),np.cos(k*xx)])
   r=((xx[:,None,:]-xx[None,:,:])**2).sum(2)/xx.shape[1]; ks[coord+'rbf'+str(k)]=np.exp(-k*r)
   ks[coord+'add'+str(k)]=np.exp(-k*(xx[:,None,:]-xx[None,:,:])**2).mean(2)
 growth=np.c_[z,z*z,z**3,z**4,z**5,np.sin(2.5*z),np.sqrt(.08+z*z),np.maximum(z[:,[1,3]],0)]; smooth=kernel(growth); alphas=np.logspace(-8,-.7,23); target=y[:n]-y[:n].mean()
 for nm,K in ks.items():
  for fac in [0,.003,.03,.3,3]:
   if fac==0 and 'cross' in nm:continue
   M=K+fac*smooth+.05*cross; ev,U=eigh(M[:n,:n]); E=1/np.maximum(ev[:,None]+alphas,1e-12); coef=U@((U.T@target)[:,None]*E); looerr=np.mean(abs(coef/(U**2@E)),axis=0); preds=y[:n].mean()+M[n:,:n]@coef; err=np.mean(abs(preds-y[n:,None]),axis=0)
   for i in np.argsort(looerr)[:2]:out.append((looerr[i],err[i],nm,fac,alphas[i],preds[:,i]))
 print(label,'LOO',[(a,b,c,d,e) for a,b,c,d,e,_ in sorted(out,key=lambda a:a[0])[:12]],flush=True);print(label,'CV',[(a,b,c,d,e) for a,b,c,d,e,_ in sorted(out,key=lambda a:a[1])[:12]],flush=True)
 # Low-dimensional generalized additive variable projection, multi-start.
 A=poly(z);o=[]; rng=np.random.default_rng(32)
 for typ in ['tanh','soft','sin','rat','mix','one']:
  def F(w):
   k=np.exp(w[:4]); v=(z-w[4:])*k
   H={'tanh':lambda:np.tanh(v),'soft':lambda:np.sqrt(1+v*v),'sin':lambda:np.sin(v),'rat':lambda:v/(1+v*v),'mix':lambda:np.c_[np.sin(v[:,[0,2]]),np.sqrt(1+v[:,[1,3]]**2),z[:,3]**3], 'one':lambda:np.c_[np.sqrt(1+v[:,[1,3]]**2),np.tanh(v[:,[0,2]]),z[:,3]**3]}[typ]()
   return np.c_[A,H]
  def res(w,ret=False):
   d=F(w); Q,R=np.linalg.qr(d[:n]);fit=ls(d[:n],y[:n]);e=d[:n]@fit-y[:n]; lever=(Q**2).sum(1);return (np.mean(abs(e/(1-lever))), np.mean(abs(d[n:]@fit-y[n:])),w) if ret else np.r_[e/np.sqrt(np.maximum(.01,1-lever)),.002*w]
  opts=[]
  for r in range(15):
   init=np.r_[rng.uniform(-.3,1.2,4),rng.normal(0,.24,4)]; w=least_squares(res,init,bounds=([-1.4]*4+[-1]*4,[2.15]*4+[1]*4),max_nfev=160).x;opts.append(res(w,True))
  opts.sort(key=lambda a:a[0]); print('add',label,typ,opts[0], 'minCV',min(o[1] for o in opts),flush=True)
run(z,b,p,q,'refined');
exec(open('/tmp/latent.py').read().split("print('cal'")[0]);b,p,q=ops(s,z);run(z,b,p,q,'bilinear')
