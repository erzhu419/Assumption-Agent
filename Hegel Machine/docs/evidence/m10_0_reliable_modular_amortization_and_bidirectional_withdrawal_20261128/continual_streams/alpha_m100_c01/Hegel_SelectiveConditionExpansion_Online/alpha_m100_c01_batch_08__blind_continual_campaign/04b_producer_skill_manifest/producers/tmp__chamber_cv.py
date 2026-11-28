exec(open('/tmp/probe_chamber.py').read().split('for j,xq in enumerate(xqs):')[0])
from scipy.optimize import least_squares
Q=d['chamber_verification_trials']; labs=Q['lab_id']; missing=np.isnan(Q['coupon_features_raw']).any(1)
base=np.column_stack((h[:,None]*th[:2],np.zeros(len(h))))@B.T
R=Y-base
linear=np.r_[t,1.]; decay=lambda s:np.r_[-s*np.expm1(-t/s),1.]; F,S=map(decay,taus)
models={
 'linear':([], lambda a:linear, ([],[])),
 'one_decay':([1.2],lambda a:decay(np.exp(a[0])),([-4],[5])),
 'one_decay_remainder':([.3,.45],lambda a:linear+a[1]*(decay(np.exp(a[0]))-linear),([-4,0],[5,1])),
 'shared_modes_remainder':([.15,.55],lambda a:linear+a[0]*(F-linear)+a[1]*(S-linear),([0,0],[1,1])),
 'rational_integral':([-1],lambda a:np.r_[np.log1p(t*np.exp(a[0]))/np.exp(a[0]),1.],([-6],[4])),
 'rational_strain':([-2],lambda a:np.r_[t/(1+t*np.exp(a[0])),1.],([-6],[4])),
 'rate_inverse_square_interp':([-1,.6],lambda a:linear+a[1]*(np.r_[t/(1+t*np.exp(a[0])),1.]-linear),([-6,0],[4,1])),
}
for mode,xq in zip(('joint','train_q','diag'),xqs):
 drift=f(xq)@th[2:]
 print('NORMALIZATION',mode,'rate residual by lab',[(int(l),np.round((R[:,-1]-drift)[labs==l],4)) for l in np.unique(labs)])
 for name,(a,func,bounds) in models.items():
  def fit(rows):
   if not len(a): return a
   return least_squares(lambda u:(R[rows]-drift[rows,None]*func(u)).ravel(),a,bounds=bounds,loss='soft_l1',f_scale=.005).x
  E=np.empty_like(R)
  for l in np.unique(labs):
   rows=labs==l; E[rows]=R[rows]-drift[rows,None]*func(fit(~rows))
  out=lambda e:np.round([np.sqrt(np.mean(e**2)),np.quantile(abs(e),.9),np.max(abs(e))],4)
  print(name,np.round(fit(np.ones(len(labs),bool)),5),'curve CV all/complete',out(E[:,:-1]),out(E[~missing,:-1]),'rate CV',out(E[:,-1]),'worst lab curve/rate RMSE',max(out(E[labs==l,:-1])[0] for l in np.unique(labs)),max(out(E[labs==l,-1])[0] for l in np.unique(labs)))
print('missing rows:',np.flatnonzero(missing))
