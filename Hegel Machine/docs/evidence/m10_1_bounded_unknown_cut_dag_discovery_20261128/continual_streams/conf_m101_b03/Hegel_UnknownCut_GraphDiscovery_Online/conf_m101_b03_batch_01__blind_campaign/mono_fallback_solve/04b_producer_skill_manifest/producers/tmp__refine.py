import pickle,numpy as np
from scipy.optimize import least_squares
from importlib.machinery import SourceFileLoader
from latent import reverse_a
from operators import poly
from compact_c import recover

def panel(p, k, mode):
 u=p['u']; v=p[k]; fit=np.linalg.lstsq
 def design(x,h):
  a=poly(x,1)
  if k=='v1':
   w=h[0]; f=np.tanh(w*x)
   if mode==1: f=f+h[1]*x**3
   if mode==2: f=f+h[1]*np.sin(h[2]*x)
   if mode==3: f=f+h[1]*x/(1+(h[2]*x)**2)
   if mode==4: f=f+h[1]*x*x
   if mode==5: f=f+h[1]*np.cos(h[2]*x)
  else:
   w=h[0]; f=x*x+h[1]*(np.cos(w*x)-1) if mode==0 else x*x+h[1]*(1/(1+(w*x)**2)-1)
  return np.c_[a,f]
 def residual(h,loo=False):
  d=design(u,h); q,_=np.linalg.qr(d); r=v-q@(q.T@v)
  return r/(1-(q*q).sum(1,keepdims=True)) if loo else r
 if k=='v1':
  starts={0:[[2.3]],1:[[2.3,0]],2:[[w,c,j] for w in [1,2.3,3] for c in [-.3,.3] for j in [1.5,3]],3:[[w,c,j] for w in [1,2.3,3] for c in [-.3,.3] for j in [.8,2]],4:[[2.3,0]],5:[[2.3,.05,j] for j in [1,2,3]]}[mode]
 else: starts=[[w,-.2] for w in [1,2.3,3]] if mode==0 else [[w,-1] for w in [.5,1,2]]
 opts=[least_squares(lambda h:residual(h).ravel(),start,max_nfev=300) for start in starts]
 r=min(opts,key=lambda r:np.mean(residual(r.x,True)**2)); d=design(u,r.x); b=fit(d,v,rcond=None)[0]; e=np.sqrt(np.mean(residual(r.x,True)**2)); print(k,mode,r.x.round(5),round(e,7)); return e, lambda x:design(x,r.x)@b

if __name__=='__main__':
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; z=reverse_a(s['panel_a'],tr['x']); p=s['panel_c'];
 c1=min([panel(p,'v1',i) for i in range(6)],key=lambda t:t[0]); c2=min([panel(p,'v2',i) for i in range(2)],key=lambda t:t[0]);
 v,w=c1[1](z),c2[1](z); D=np.c_[poly(z,1),v,w,v*w,v*v,w*w]; prior=SourceFileLoader('prior','/tmp/hegel_continual_archive/conf_m101_b03_batch_00_solver.py').load_module();
 for n in [52,56]:
  H,M=prior._time_modes(s['eval_times'],tr['traj'][:n]);Y=np.c_[tr['y'][:n],M]; b=np.linalg.lstsq(D[:n],Y,rcond=None)[0]; O=D[n:]@b; print(n,'MAE',np.mean(abs(O[:,0]-tr['y'][n:])),np.mean(abs(O[:,1:]@H.T-tr['traj'][n:])), 'train',np.mean(abs(D[:n]@b-Y),0));print('coefs',b.T.round(4))
  for name,F in [('add1',D[:,:13]),('cross',D[:,:17]),('products',D),('poly+C',np.c_[poly(z,2),D[:,5:]]),('mixed',np.c_[D,z*v,z*w])]:
   for lam in [0,.0003,.003,.03]:
    a=F[:n]; bb=np.linalg.solve(a.T@a+lam*np.eye(a.shape[1]),a.T@Y); O=F[n:]@bb; print(name,lam,np.round([np.mean(abs(O[:,0]-tr['y'][n:])),np.mean(abs(O[:,1:]@H.T-tr['traj'][n:])),np.mean(abs(O[:,1:]@H[-1]-tr['traj'][n:,-1]))],5))
