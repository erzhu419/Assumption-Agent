import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/chamber98.py').read())
from scipy.optimize import least_squares
one=np.ones(len(H)); X=m._features(a,c)[n:]; drift=th[2]+X@th[3:]; V=np.c_[Y,2*R]
def basis(ta):
 return np.vstack([np.c_[-np.expm1(-t[:,None]/ta),t],np.r_[1/ta,1]*2])
for slow in [2.4,2.6,3.0,3.4,4.38]:
 ta=np.array([tau[0],slow]); T=basis(ta); Q=np.linalg.lstsq(T,V.T,rcond=None)[0].T
 print('slow tau',slow,'temporal rms',np.std(V-Q@T.T),'gap errors vs hold only',np.round(np.sqrt(np.mean((Q[:,:2]/th[:2]-H[:,None])**2,0)),4),'drift slope/intercept',np.round(np.polyfit(drift,Q[:,2],1),4)); print(np.round(np.c_[Q[:,:2]/th[:2]-H[:,None],Q[:,2]],3).T)
def mem(ta):
 return P@(-np.expm1(-dt[:,None]/ta)*np.exp(-later[:,None]/ta))
def fields(q,mode):
 ta=tau*np.exp(q); z=mem(ta); gap=H[:,None]-z
 if mode=='reset': return [H[:,None],H[:,None],drift[:,None]]
 if mode=='reset_offset': return [np.c_[H,one],np.c_[H,one],np.c_[drift,one]]
 if mode=='memory': return [np.c_[H,z[:,0]],np.c_[H,z[:,1]],drift[:,None]]
 if mode=='intrinsic_memory': return [np.c_[H,Z[:,0]],np.c_[H,Z[:,1]],drift[:,None]]
 if mode=='reset_history': return [np.c_[H,P],np.c_[H,P],drift[:,None]]
 if mode=='reset_nonlinear': return [np.c_[H,H**2],np.c_[H,H**2],drift[:,None]]
 if mode=='benchtop_gap': return [gap[:,[0]],gap[:,[1]],drift[:,None]]
 if mode=='strain0_reset': return [np.c_[H,c['strain_0']],np.c_[H,c['strain_0']],drift[:,None]]
def design(q,mode):
 T=basis(tau*np.exp(q)); fs=fields(q,mode); return np.concatenate([T[None,:,k,None]*F[:,None,:] for k,F in enumerate(fs)],axis=2)
for mode in ['reset','reset_offset','memory','intrinsic_memory','reset_history','reset_nonlinear','benchtop_gap','strain0_reset']:
 for fixed in [True,False]:
  def calc(mask):
   def profile(q,ret=False):
    D=design(q,mode); F=D[mask].reshape(-1,D.shape[-1]); w=np.linalg.lstsq(F,V[mask].ravel(),rcond=None)[0]; pred=D@w
    return (pred,w) if ret else (pred[mask]-V[mask]).ravel()
   q=np.zeros(2) if fixed else least_squares(profile,[0,-.5],bounds=([-.7,-1.5],[.6,1.2])).x
   return q,*profile(q,True)
  q,pred,w=calc(np.ones(len(H),bool)); rms=np.sqrt(np.mean((pred-V)**2,0)); oo=np.empty_like(V)
  for l in np.unique(lab):
   mask=lab!=l; oo[~mask]=calc(mask)[1][~mask]
  worst=np.max([np.sqrt(np.mean((oo[lab==l]-V[lab==l])**2,0)) for l in np.unique(lab)],0)
  print(mode,'fixed',fixed,'tau',np.round(tau*np.exp(q),4),'coeff',np.round(w,5),'fit',np.round([np.mean(rms[:-1]**2)**.5,rms[-1]/2],5),'worst',np.round([np.mean(worst[:-1]**2)**.5,worst[-1]/2],5))
