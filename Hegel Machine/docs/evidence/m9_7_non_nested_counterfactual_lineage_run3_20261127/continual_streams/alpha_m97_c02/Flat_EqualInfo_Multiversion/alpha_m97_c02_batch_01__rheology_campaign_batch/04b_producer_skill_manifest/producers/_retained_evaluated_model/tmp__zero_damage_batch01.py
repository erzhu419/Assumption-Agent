exec(open('/tmp/check_batch01.py').read().split('out = m.solve')[0])
y = train['strain_curves_obs'] - train['strain_0'][:,None]
r = train['instant_rate_obs']
t = data[-1]
print('times',t)
wref = np.r_[gt['theta'],gt['vp_theta']]
c,a = m._design(*data,gt['taus'])
mask = ~data[3][n:] & np.isfinite(test['coupon_features_raw']).all(1)
o = gt['te_clean'] - test['strain_0'][:,None]
rate_error = gt['te_true_rate0'] - a[n:] @ wref
base = c[n:] @ wref
b = wref[2] + data[2][n:] @ wref[3:7]
for name, phi in [('linear',t),('log',np.log1p(t)),('sqrt',2*(np.sqrt(1+t)-1)),('exp',-np.expm1(-t)),('sat',t/(1+t)),('asinh',np.arcsinh(t))]:
    e=o-base-b[:,None]*(phi-t)-rate_error[:,None]*phi
    print(name,'zero damage diagnostic RMSE',np.sqrt(np.mean(e[mask]**2)))
known_prony = c[n:,:,:2] @ wref[:2]
true_drift = b + rate_error
ratio = (o-known_prony)[mask] / true_drift[mask,None]
print('implied terminal function median',np.median(ratio,axis=0),'spread',np.ptp(ratio,axis=0))
complete=np.isfinite(train['coupon_features_raw']).all(1)
for label, mask in [('active-complete',data[3][:n]&complete),('active-all',data[3][:n])]:
    taus,w=m._fit(tuple(x[:n][mask] for x in data[:5])+data[5:],y[mask],r[mask]); print(label,taus,w)
