import numpy as np
from scipy.optimize import least_squares
z=np.load('/tmp/chamber_explore.npz'); tau,th,dt,t=[z[k] for k in ('tau','theta','dt','tt')]
tr={k[3:]:z[k] for k in z.files if k.startswith('tr_')}; te={k[3:]:z[k] for k in z.files if k.startswith('te_')}; q={k[2:]:z[k] for k in z.files if k.startswith('q_')}
def group(part): return np.column_stack((part['lab_id'],part['stratum']))
lab=np.r_[tr['lab_id'],te['lab_id']]; ref=np.r_[tr['ref_coupons'],te['ref_coupons']]; x=q['coupon_features_raw'].copy()
for i in np.unique(lab):
 p=ref[lab==i]; x[q['lab_id']==i]=(x[q['lab_id']==i]-np.median(p.mean(0)))/np.median(p.std(0))
_,j,n=np.unique(np.r_[group(tr),group(te)],axis=0,return_inverse=True,return_counts=True); xx=np.r_[z['x'],z['xt']]; w=1./n[j]; x=np.where(np.isnan(x),w@xx/w.sum(),x)
p=q['past_channel_raw'].sum(-1); h=q['hold_channel_raw'].sum(-1); f=th[2]+x@th[3:]; hist=p@(np.exp(-np.r_[np.cumsum(dt[:0:-1])[::-1],0.][:,None]/tau)*-np.expm1(-dt[:,None]/tau)); y=np.c_[q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs']]; B=np.r_[np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1][None,:]]
base=np.c_[(h[:,None]-hist)*th[:2],f]; old=np.c_[h[:,None]*th[:2]+ f[:,None]*tau*[.45,.52],.03*f]; reset=np.c_[h[:,None]*th[:2],f];
for name,a in [('base',base),('old',old),('reset',reset)]:
 e=a@B.T-y; print(name,'med',np.median(abs(e),axis=0).round(4),'signed med',np.median(e,axis=0).round(4))
a=np.array([least_squares(lambda a:B@a-yi,np.linalg.lstsq(B,yi,rcond=None)[0],loss='soft_l1',f_scale=.012).x for yi in y]); print('p, h, f, x, a-h*th, drift',np.c_[p,h,f,x,a[:,:2]-h[:,None]*th[:2],a[:,2]].round(3)); print('row param fit err',np.quantile(abs(a@B.T-y),[.5,.9,.98]));
raw=np.array([np.r_[1,hi,fi,xi,pi,hi*fi,fi**2,hi**2] for hi,fi,xi,pi in zip(h,f,x,p)])
print('corr with inferred corrections',np.corrcoef(np.c_[raw[:,1:],a-reset].T)[-3:,:-3].round(2)); np.savez('/tmp/chamber_params.npz',B=B,y=y,base=base,old=old,reset=reset,p=p,h=h,f=f,x=x,a=a,hist=hist,lab=q['lab_id'],tau=tau,th=th,dt=dt,t=t)
