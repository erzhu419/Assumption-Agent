exec(open('/tmp/analyze_chamber.py').read().split('for s,xx in ')[0])
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True)
y=_response(dg); h=dg['hold_channel_raw'].sum(-1); p=dg['past_channel_raw'].sum(-1); drift=th[2]+xd@th[3:]; n=len(h)
print('metadata', d.keys(),dg.keys(), 'diag labs',np.bincount(dg['lab_id']));print('times',t,'dt',dt)
def basis(q): return np.vstack((np.r_[-np.expm1(-t[:,None]/q),[1/q]],np.r_[t,1.][:,None].T)) if False else np.column_stack((np.vstack((-np.expm1(-t[:,None]/q),1/q)),np.r_[t,1.]))
def co(q): return np.linalg.lstsq(basis(q),y.T,rcond=None)[0].T
q=least_squares(lambda q:(co(q)@basis(q).T-y).ravel(),tau).x
v=co(q); d0=v[:,2]
u=dt[:,None]/tau; mem=p@(-np.expm1(-u)*np.exp(-np.cumsum(u[::-1],axis=0)[::-1]+u));ones=np.ones(n)
print('q',q); print('first versus h',np.c_[h,drift,v[:,0]-th[0]*h,v[:,1]-th[1]*h,d0]);
models={'reset+drift':np.c_[h,drift],'reset+drift+constant':np.c_[h,drift,ones],'reset+effective drift':np.c_[h,d0], 'reset+effective drift+constant':np.c_[h,d0,ones], 'memory+drift':np.c_[h,drift,mem], 'history+drift':np.c_[h,drift,p], 'reset+free feature weights':np.c_[h,ones,xd], 'reset+measured telemetry':np.c_[h,drift,ones,dg['aux_chamber_telemetry']] if 'aux_chamber_telemetry' in dg else np.c_[h,drift]}
for name,a in models.items():
 b=np.linalg.lstsq(a,v,rcond=None)[0]; err=v-a@b; out=[]
 for j in range(n):
  mask=np.arange(n)!=j; out.append((a[j]@np.linalg.lstsq(a[mask],v[mask],rcond=None)[0])@basis(q).T-y[j])
 out=np.asarray(out);print(name,'coefficients',b,'component max',abs(err).max(0),'LOCO p90',np.quantile(abs(out),.9,axis=0)[[0,-2,-1]])
print('diagnostics feature missing mask',np.argwhere(~np.isfinite(dg['coupon_features_raw'])).tolist()); print('train/test/diag medians/stds of reference means/stds by lab');
for s in (tr,te,dg):
 print([(int(j),np.median(s['ref_coupons'][s['lab_id']==j].mean(0)),np.median(s['ref_coupons'][s['lab_id']==j].std(0))) for j in np.unique(s['lab_id'])]);
