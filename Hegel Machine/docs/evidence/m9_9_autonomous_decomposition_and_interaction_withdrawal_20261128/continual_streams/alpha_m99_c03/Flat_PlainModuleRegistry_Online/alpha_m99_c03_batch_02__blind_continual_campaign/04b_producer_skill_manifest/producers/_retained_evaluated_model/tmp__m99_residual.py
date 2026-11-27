from m99_probe import *
from scipy.optimize import least_squares, minimize_scalar
n=len(tr['lab_id']); bridge=m._calibration(cal); X=m._features(tr,te); ct=m._coupon_inputs(tr,X[:n],bridge); taus,theta=m._fit(ct,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs'])
Y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]; print('fields',[(k,np.shape(v)) for k,v in g.items()]); print('test fields',list(te)); print('lab counts',np.unique(g['lab_id'],return_counts=True))
def G_using(blocks):
 lab=np.concatenate([b['lab_id'] for b in blocks]); strata=np.concatenate([b['stratum'] for b in blocks]); ref=np.vstack([b['ref_coupons'] for b in blocks]); X=np.vstack([b['coupon_features_raw'] for b in blocks]).copy(); G=g['coupon_features_raw'].copy();
 for l in np.unique(lab):
  p=ref[lab==l]; c=np.median(p.mean(0)); s=np.median(p.std(0)); X[lab==l]=(X[lab==l]-c)/s; G[g['lab_id']==l]=(G[g['lab_id']==l]-c)/s
 _,ix,k=np.unique(np.c_[lab,strata],axis=0,return_inverse=True,return_counts=True); w=1/k[ix]; col=np.nansum(X*w[:,None],0)/np.sum(np.isfinite(X)*w[:,None],0); return np.where(np.isnan(G),col,G)
def rms(a): return np.sqrt(np.mean(a*a))
for desc,G in [('tr+g',m._features(tr,g)[n:]),('tr+te',G_using([tr,te])),('tr+te+g',G_using([tr,te,g]))]:
 cg=m._coupon_inputs(g,G,bridge); P,H,D=cg; F,R=m._design(taus,cg,dt,t); E=Y-np.c_[F@theta,R@theta]; drift=D@theta[2:]; U,S,V=np.linalg.svd(E,full_matrices=False); print('\nnormalization',desc,'svd',S.round(4),'shape1',V[0].round(4),'shape2',V[1].round(4)); print('missing',np.argwhere(np.isnan(g['coupon_features_raw'])).tolist());
 aux=g.get('aux_chamber_telemetry',np.empty((len(G),0))); Zs={'r':drift[:,None],'1+r':np.c_[np.ones(len(G)),drift],'X':D,'X+stress':np.c_[D,H,P], 'X+aux':np.c_[D,aux], 'X+aux+stress':np.c_[D,aux,H,P]}; A=np.c_[D[:,1:],H,P,aux]; print('aux example',aux[:3]); print('corr with shapes',np.corrcoef(np.c_[A,U[:,:3]].T)[-3:,:-3].round(2));
 T=np.r_[t,1.];
 def exp(u): return np.r_[-np.expm1(-t/u),1/u]
 def shape(u,mode):
  if mode=='exp+linear': return np.array([exp(u),T]);
  if mode=='exp-drift': return np.array([u*exp(u)-T]);
  if mode=='two-time': return np.array([exp(taus[0]),exp(u),T]);
 for mode in ('exp+linear','exp-drift','two-time'):
  for name in ('r','1+r','X','X+aux'):
   Z=Zs[name]; Q=np.linalg.svd(Z,full_matrices=False)[0][:,:np.linalg.matrix_rank(Z)]; C=Q.T@E
   def fit(u):
    B=shape(u,mode); coef=np.linalg.lstsq(B.T,C.T,rcond=None)[0].T; return Q@coef@B, np.linalg.norm(C-coef@B)**2
   fitu=minimize_scalar(lambda u:fit(u)[1],bounds=(.15,5),method='bounded').x; pred=fit(fitu)[0]; r=E-pred; print(mode,name,round(fitu,5),'rmse',[round(rms(r[:,:-1]),5),round(rms(r[:,-1]),5)],'unit r e.g.',np.linalg.lstsq(Z,pred,rcond=None)[0][-1,[0,4,-2,-1]].round(4))
 if desc=='tr+te':
  print('row first,late,rate residual & drift'); print(np.c_[E[:,[0,-2,-1]],drift].round(4));
