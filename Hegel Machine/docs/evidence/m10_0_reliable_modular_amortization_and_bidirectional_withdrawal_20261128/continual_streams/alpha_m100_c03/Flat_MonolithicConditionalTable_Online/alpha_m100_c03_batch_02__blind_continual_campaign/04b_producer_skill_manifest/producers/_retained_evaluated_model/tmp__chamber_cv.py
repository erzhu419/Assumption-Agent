from chamber_probe import *
from itertools import combinations
labs=di['lab_id']; n=len(labs); K=len(ts)
# Baseline predictions are fixed by the material fit on benchtop coupons.
base=np.c_[A@w,R@w]; target=np.c_[y,r]; residual=target-base
print('per-coupon [g, residual-at-last-time, residual-initial-rate, missing]')
print(np.round(np.c_[g,e[:,-1],er,np.isnan(di['coupon_features_raw']).any(axis=1)],4))
print('channel residual scales by time and rank',np.round(np.median(abs(residual),axis=0),5),np.round(np.linalg.svd(e,compute_uv=False),4))
for name, panel in [('train',tr),('test',te),('diag',di)]:
 print(name,[(int(l),round(float(np.median(panel['ref_coupons'][panel['lab_id']==l].mean(0))),4),round(float(np.median(panel['ref_coupons'][panel['lab_id']==l].std(0))),4)) for l in np.unique(panel['lab_id'])])

def basis(t):
 return np.r_[-np.expm1(-ts/t),1/t]
lin=np.r_[ts,1.]; cols={'g':g[:,None],'ag':np.c_[np.ones(n),g],'gx':np.c_[g,x[:, :]],'g2':np.c_[g,g*g],'gapg':np.c_[g,gap],'allx':np.c_[np.ones(n),x]}

def run(mode, name, idx):
 f=cols[name]
 def design(q):
  tt=np.exp(q[0]) if len(q) else 1.
  if mode=='ramp': B=np.array([lin-tt*basis(tt)])
  elif mode=='free': B=np.array([lin,basis(tt)])
  elif mode=='intrinsic': B=np.array([lin,basis(tau[0]),basis(tau[1])])
  elif mode=='two_ramps': B=np.array([lin-tt*basis(tt),lin-np.exp(q[1])*basis(np.exp(q[1]))])
  elif mode=='intrinsic_rate': B=np.array([lin-tau[0]*basis(tau[0]),lin-tau[1]*basis(tau[1])])
  return (f[:,None,:,None]*B.T[None,:,None,:]).reshape(n,K+1,-1)
 q0={'ramp':[0.], 'free':[0.],'intrinsic':[],'intrinsic_rate':[], 'two_ramps':[-.4,.7]}[mode]
 b=design(q0); scale=np.r_[np.full(K,.012),.012]; z=b[idx]/scale[None,:,None]; v=residual[idx]/scale
 c0=np.linalg.lstsq(z.reshape(-1,z.shape[-1]),v.ravel(),rcond=1e-9)[0]
 lb=np.r_[np.full(len(q0),np.log(.08)),np.full(len(c0),-np.inf)]; ub=np.r_[np.full(len(q0),np.log(15)),np.full(len(c0),np.inf)]
 def fun(p):return ((design(p[:len(q0)])[idx]@p[len(q0):]-residual[idx])/scale).ravel()
 fit=least_squares(fun,np.r_[q0,c0],bounds=(lb,ub),loss='soft_l1',f_scale=1,max_nfev=160)
 return base+design(fit.x[:len(q0)])@fit.x[len(q0):],fit.x

def score(pred):
 es=pred-target; out=[]
 for ind in [slice(None),slice(-4,-1),-1]:
  z=es if ind==slice(None) else es[:,ind];z=z[:,:-1] if ind==slice(None) else z
  scores=[np.sqrt(np.mean(z[labs==l]**2)) for l in np.unique(labs)]; out.extend([np.median(abs(z)),np.sqrt(np.mean(z*z)),max(scores)])
 return np.round(out,5)
print('metrics triplets median/RMSE/worst-lab RMSE by [curve,late,rate]');print('baseline',score(base))
for mode,name in [('ramp','g'),('free','g'),('intrinsic','g'),('intrinsic_rate','g'),('two_ramps','g'),('ramp','ag'),('free','ag'),('free','g2'),('ramp','allx'),('free','allx'),('free','gapg')]:
 pred=np.empty_like(base)
 for l in np.unique(labs):
  z,_=run(mode,name,labs!=l); pred[labs==l]=z[labs==l]
 full,par=run(mode,name,np.ones(n,bool));print(mode,name,score(pred),'pars',np.round(par,6))
