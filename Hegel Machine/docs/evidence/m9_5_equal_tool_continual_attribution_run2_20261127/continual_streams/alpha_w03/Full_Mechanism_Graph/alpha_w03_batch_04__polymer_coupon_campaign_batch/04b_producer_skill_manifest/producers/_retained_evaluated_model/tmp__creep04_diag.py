exec(open('/tmp/creep04_oof.py').read().split('for linear in')[0])
import json
history=json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))
for tab in history+[d['calibration_load_cells']]:
 print('sweep', [None if (b:=s._supported_line(*s._checks(tab,l))) is None else b.round(5).tolist() for l in range(4)])
curve=np.empty_like(y);rate=np.empty_like(r);coef=np.empty((n,9)); imputed=np.isnan(tr['coupon_features_raw'])
for ell in sorted(set(lab)):
 a=lab!=ell;b=~a
 tau,w=s._fit(x[a],h[a],p[a],dt,t,hold[a],work[a],y[a],r[a]);c,v,*_=s._design(tau,x[b],h[b],p[b],dt,t,hold[b],work[b]);curve[b]=c@w;rate[b]=v@w;coef[b]=w
for label,m in [('complete',~miss),('missing',miss),('missing wing A',miss&(lab<2)),('missing wing B',miss&(lab>=2))]:
 print(label,'n',m.sum(),'rate abs quantiles',np.quantile(abs(rate[m]-r[m]),[.5,.8,.95,1]).round(5))
 for alpha in (0,.8,.95,1):
  e=curve[m]-y[m]+alpha*(r[m]-rate[m])[:,None]*t
  print(alpha,'curve abs p50,p80,p90,p95,max',np.quantile(abs(e),[.5,.8,.9,.95,1]).round(5),'capped RMSE',np.sqrt(np.mean(np.minimum(e**2,.05**2),axis=0)).round(5))
print('complete max per lab',[(int(i),round(np.max(abs(rate[(lab==i)&~miss]-r[(lab==i)&~miss])),5)) for i in np.unique(lab)])
print('missing rate correction candidates',np.c_[lab[miss],imputed[miss].astype(int),np.round((r-rate)[miss],4)])
