exec(open('/tmp/chamber_probe.py').read().split('for f in ([.0')[0])
from collections import defaultdict
from itertools import product
labs=q['lab_id']; err=y-_design(q,xq,dt,t,tau)@b
labm=[labs==j for j in np.unique(labs)]
ps=np.r_[np.zeros(6),1]; lag=np.cumsum(dt[::-1])[::-1]-dt
E=lambda tt: np.vstack((-np.expm1(-t[:,None]/tt),1/tt)).T
D=lambda v: np.asarray(v)[:,None]
H=lambda v: D(v)*time
W=lambda v: D(v)*np.r_[t*t,0.]
base0=_design(q,xq,dt,t,tau)@b
features={'steady':H(drift),'f0':ds[:,0]-steady,'f1':ds[:,1]-steady,'q':W(drift),'sq':W(drift**2),'hqd':W(h*drift),'1q':W(h*0+1),'t2':np.broadcast_to(np.r_[t*t,0.],y.shape),'m0':mem[:,0],'m1':mem[:,1],'h0':hs[:,0],'h1':hs[:,1],'d2':H(drift**2),'d3':H(drift**3),'drift-offset':H(drift*0+1),'d-rate':D(drift)*np.r_[np.zeros(len(t)),1.]}
def basis(names):return np.stack([features[k] for k in names],axis=-1)
def summarize(label,names,baseline=base0,nonlin=None):
 a=basis(names) if names else np.empty((*y.shape,0));
 def fit(m):
  aa=a[m].reshape(m.sum()*y.shape[1],len(names)); yy=(y-baseline)[m].reshape(-1)
  init=np.linalg.lstsq(aa,yy,rcond=None)[0]; p0=init if nonlin is None else np.r_[init,0.]
  if not len(p0):return p0
  def f(p):
   r=aa@p[:len(names)]-yy
   if nonlin is not None:r=r+nonlin(p[-1])[m].ravel()
   return r
  return least_squares(f,p0,loss='soft_l1',f_scale=.014,max_nfev=120).x
 def pred(p):return baseline+a@p[:len(names)]+(0 if nonlin is None else nonlin(p[-1]))
 w=fit(np.ones(len(y),bool)); loo=np.zeros_like(y)
 for m in labm:loo[m]=(y-pred(fit(~m)))[m]
 score=np.array([[np.mean(abs(loo[m,:3])),np.mean(abs(loo[m,-4:-1])),np.mean(abs(loo[m,-1]))] for m in labm]); med=np.median(abs(y-pred(w)),axis=0)
 print(label,'coef',np.round(w,4),'worst early late rate',np.round(score.max(0),4),'pooled',np.round([np.mean(abs(loo[:,:-1])),np.mean(abs(loo[:,-1]))],4),'in-sample endpoint/rate med',np.round(med[[-2,-1]],4))
 return w,loo
summarize('benchtop',[])
basekeys=['f0','f1','steady']; summarize('transient + free steady',basekeys); summarize('transient + free steady + memory',basekeys+['m0','m1']); summarize('transient + free steady + hold/memory',basekeys+['m0','m1','h0','h1'])
for k in ['q','sq','hqd','1q','d2','d3','drift-offset','d-rate']:
 summarize('transient/steady + '+k,basekeys+[k])
for k in range(6):
 for b0 in range(2): features[f'p{k}b{b0}']=D(past[:,k])*g[b0]
for k in range(6):summarize(f'transient/steady + past[{k}]',basekeys+[f'p{k}b0',f'p{k}b1'])
for kind in ['faster memory','faster chamber','drift saturation']:
 def f(z0):
  v=np.exp(np.clip(z0,-3,3))
  if kind=='drift saturation':return H(drift/(1+v*drift**2)-drift) if False else D(drift)*np.r_[-np.expm1(-v*t)/v-t,0.]
  tt=tau/v; u0=dt[:,None]/tt; state=past@(-np.expm1(-u0)*np.exp(-lag[:,None]/tt));
  if kind=='faster memory':return ((z-state)*b[:2])@g
  return ((h[:,None]-state)*b[:2])@E(tt)-((h[:,None]-z)*b[:2])@g
 if kind!='drift saturation':summarize(kind,basekeys,nonlin=f)
# Compare endpoint residuals by coupon to identify which input drives remaining lab-level error.
w,e=summarize('reference',basekeys)
print('per coupon lab/stratum/missing/drift, rate and final OOF errors'); print(np.round(np.column_stack((labs,q['stratum'],np.isnan(q['coupon_features_raw']).sum(1),drift,e[:,-1],e[:,-2])),3))
r=d['calibration_load_cells']; print('canonical bridge exact-match rows',np.count_nonzero(np.all(r['bridge_raw']==r['true_channel_loads'],axis=1)),'raw inlier fit max nominal-payload delta',np.max(abs(np.linalg.lstsq(np.stack([raw['calibration_load_cells']['true_channel_loads'][:,0],np.ones(len(raw['calibration_load_cells']['true_channel_loads']))],1)[np.all(abs(raw['calibration_load_cells']['bridge_raw'])<=8.5,axis=1)],raw['calibration_load_cells']['bridge_raw'][np.all(abs(raw['calibration_load_cells']['bridge_raw'])<=8.5,axis=1),0],rcond=None)[0]-np.array([pay['gains'][0],pay['offsets'][0]]))))
