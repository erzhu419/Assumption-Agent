exec(open('/tmp/probe98.py').read().split('for ref in (g,')[0])
from scipy.optimize import least_squares
x=m._features(g,g)[-len(hold):]; drift=np.c_[np.ones(len(hold)),x]@th[2:]
ex=np.r_[-np.expm1(-t[:,None]/tau),[1/tau]]
A=m._design((past,hold,np.c_[np.ones(len(hold)),x]),dt,t,tau)
elastic=hold[:,None]*(ex@th[:2]); y=gy-elastic; lin=np.r_[t,1.]; hist=A[:,:,:2]@th[:2]-elastic
cal=d['calibration_load_cells']; print('calibration total error quantiles',np.quantile(abs((cal['bridge_raw']-o)@c-cal['true_channel_loads'].sum(1)),[.5,.9,1])); print('sweep shapes',{k:np.shape(v) for k,v in cal['single_channel_sweeps'].items()})
print('lab, missing, hold, drift, fitted linear drift, delta initial rate and late-minus-early drift'); print(np.c_[g['lab_id'], np.isnan(g['coupon_features_raw']).sum(1), np.round(np.c_[hold,drift,np.median(y[:,:-1]/t,axis=1),y[:,-1],np.median(y[:,7:-1]/t[7:],axis=1)-np.median(y[:,:4]/t[:4],axis=1)],3)])
scale=np.r_[np.repeat(.09,len(t)),.23]
base=(drift[:,None]*lin)
mods={
 'fixed .4':(np.zeros_like(base),.4*base), 'scaled linear':(base,0),
 'intercept+scale':(np.stack([base,np.broadcast_to(lin,base.shape)],-1),0),
 'quadratic drift':(np.stack([base,drift[:,None]*base],-1),0),
 'clipped negative':(np.stack([base,np.maximum(-drift,0)[:,None]*lin],-1),0),
 'partial memory+linear':(np.stack([base,hist],-1),0),
 'hold corr+linear':(np.stack([base,elastic],-1),0),
 'drift+linear hold':(np.stack([base,hold[:,None]*lin],-1),0),
 'drift+branch hold':(np.concatenate([base[:,:,None],hold[:,None,None]*ex],-1),0),
 'drift+branch drift':(np.concatenate([base[:,:,None],drift[:,None,None]*ex],-1),0),
 'drift+unit transient':(np.stack([base,drift[:,None]*np.r_[-np.expm1(-t),1]],-1),0),
 'hold-dependent drift':(np.stack([base,hold[:,None]*base],-1),0),
 'free linear features':(np.c_[np.ones(len(x)),x][:,None,:]*lin[None,:,None],0),
}
for name,(D,offset) in mods.items():
 if D.ndim==2:D=D[:,:,None]
 Y=y-offset
 def fit(ids):
  w=(D[ids]/scale[None,:,None]).reshape(-1,D.shape[-1]); z=(Y[ids]/scale).ravel(); p=np.linalg.lstsq(w,z,rcond=None)[0]; return least_squares(lambda q:w@q-z,p,loss='soft_l1').x
 n=len(y); ids=np.arange(n); err=np.array([D[j]@fit(ids[ids!=j])-Y[j] for j in ids]); full=fit(ids)
 print(name,np.round(full,4),'LOO curve/rate MAD and 90%',np.round([np.median(abs(err[:,:-1])),np.median(abs(err[:,-1])),np.quantile(abs(err[:,:-1]),.9),np.quantile(abs(err[:,-1]),.9)],4))
