from rheology_base import *
from scipy.optimize import least_squares
P,H,X=pq; D=X@b[2:]; Y=np.column_stack((q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs'])); w=np.r_[np.ones(len(t)),1.]; I=np.arange(len(H)); one=np.ones_like(H)
print('diagnostic keys',list(q),'groups',q['lab_id'],q['stratum']); print('H, D ranges diagnostics vs test', np.percentile(np.c_[H,D],[0,50,100],axis=0),np.percentile(np.c_[ptest[1],ptest[2]@b[2:]],[0,50,100],axis=0))
def temporal(tt): return np.row_stack((np.column_stack((-np.expm1(-t[:,None]/tt),t)), np.r_[1/tt,1]))
def memory(tt):
 z=np.zeros((len(H),len(tt)))
 for k,step in enumerate(dt):z+=-np.expm1(-step/tt)*(P[:,k,None]-z)
 return z
z=memory(tau)
Fs={'one':one,'H':H,'H2':H**2,'H3':H**3,'D':D,'D2':D**2,'tanhD':np.tanh(D),'HD':H*D,'z0':z[:,0],'z1':z[:,1],'gap0':H-z[:,0],'gap1':H-z[:,1]}
for j in range(len(dt)):Fs[f'p{j}']=P[:,j]
for j in range(1,X.shape[1]):Fs[f'x{j}']=X[:,j]
F=np.column_stack([one,H,H**3,D,D**2]); C=np.linalg.lstsq(F,Y,rcond=None)[0]; E=Y-F@C
print('time signatures [1,H,H^3,D,D^2]',np.round(C[:,[0,1,3,5,8,11,12]],4)); print('remaining corr',[(k,round(np.linalg.norm(E.T@(v-v.mean()))/(np.linalg.norm(v-v.mean())+1e-10),4)) for k,v in Fs.items()]); print('per-coupon residual rmse',np.round(np.sqrt(np.mean(E**2,axis=1)),4))
def fit(seq, idx, mode):
 def pr(ltt, ii, cf=None):
  B=temporal(np.exp(ltt)); A=np.concatenate([np.column_stack([Fs[n][ii] for n in names])[:,None,:]*B[None,:,j,None] for j,names in enumerate(seq)],axis=2); Aw=A*w[None,:,None]
  if cf is None:cf=np.linalg.lstsq(Aw.reshape(-1,Aw.shape[-1]),(Y[ii]*w).ravel(),rcond=None)[0]
  return (A@cf-Y[ii])*w,cf
 ll=np.log(tau)
 if mode=='free':ll=least_squares(lambda p:pr(p,idx)[0].ravel(),ll,bounds=(np.log([.35,1.2]),np.log([1.1,12.])),max_nfev=70).x
 return ll,pr(ll,idx)[1],pr
models=[('HD all',[['one','H','D']]*3),('drift powers',[['H','D'],['one','H3','D','D2'],['one','D','D2']]),('stress powers',[['H','D'],['one','H','H2','D'],['one','D']]),('both powers',[['one','H','D'],['one','H','H2','D','D2'],['one','H','D','D2']]),('history',[['H','D','z0'],['one','H3','D','D2','z1'],['one','D','D2']]),('linear inputs',[['one','H','D'],['one','H','D',*[f'p{j}' for j in range(6)]],['one','H','D']])]
for name,seq in models:
 for mode in ('fixed','free'):
  l,c,p=fit(seq,I,mode); e=p(l,I,c)[0]; o=[]
  for j in I:
   l2,c2,p2=fit(seq,I[I!=j],mode); o.append(p2(l2,[j],c2)[0][0])
  o=np.array(o); print(name,mode,np.exp(l).round(5),'coef',c.round(4),'fit',np.sqrt(np.mean(e[:,:-1]**2)).round(5),np.sqrt(np.mean(e[:,-1]**2)).round(5),'LOO curve/rate/worst',np.round([np.sqrt(np.mean(o[:,:-1]**2)),np.sqrt(np.mean(o[:,-1]**2)),np.max(np.sqrt(np.mean(o[:,:-1]**2,axis=1))),np.max(abs(o[:,-1]))],5))
