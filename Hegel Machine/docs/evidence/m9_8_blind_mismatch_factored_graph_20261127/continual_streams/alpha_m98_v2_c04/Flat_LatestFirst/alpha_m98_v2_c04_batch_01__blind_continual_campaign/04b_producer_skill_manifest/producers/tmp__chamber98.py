import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/probe98.py').read())
P,H=m._loads(c,B); Y=c['strain_curves_obs']-c['strain_0'][:,None]; R=c['instant_rate_obs']; lab=c['lab_id'];
U=-np.expm1(-t[:,None]/tau); A=np.vstack([np.column_stack([U,t]),np.r_[1/tau,1]*2]); target=np.column_stack([Y,2*R]); coeff=np.linalg.lstsq(A,target.T,rcond=None)[0].T
later=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; W=-np.expm1(-dt[:,None]/tau)*np.exp(-later[:,None]/tau); Z=P@W; gap=H[:,None]-Z
s=cal['single_channel_sweeps']; M=np.array([np.polyfit(s['true_stress'][s['lab_id']==l],s['cmd_stress'][s['lab_id']==l],1) for l in range(4)]); print('sweep slopes/intercepts',M)
for label,X in [('all',m._features(a,{k:np.concatenate([b[k],c[k]]) for k in c if k in b})[-len(lab):]),('joint diag',m._features(a,c)[n:]),('fixed train-test',None)]:
 if X is None:
  X=c['coupon_features_raw'].copy(); f=m._features(a,b); ll=np.r_[a['lab_id'],b['lab_id']]; rr=np.vstack([a['ref_coupons'],b['ref_coupons']]);
  for l in np.unique(lab):
   q=rr[ll==l]; X[lab==l]=(X[lab==l]-np.median(q.mean(0)))/np.median(q.std(0))
  raw=np.vstack([a['coupon_features_raw'],b['coupon_features_raw']]); fill=np.array([f[np.isnan(raw[:,j]),j][0] for j in range(4)]); X=np.where(np.isnan(X),fill,X)
 drift=th[2]+X@th[3:]; print(label,'drift per coupon:',np.round(np.column_stack([lab,drift,coeff[:,2],coeff[:,2]-drift]),3));
print('amplitudes: lab, hold, memories, gaps, inferred gaps, differences'); print(np.round(np.column_stack([lab,H,Z,gap,coeff[:,:2]/th[:2],coeff[:,:2]/th[:2]-gap]),3)); print('corr gap diff with H,Z,Past',np.round(np.corrcoef(np.column_stack([coeff[:,:2]/th[:2]-gap,H,Z,P]).T)[:2,2:],3))
print('temporal residual rms',np.sqrt(np.mean((target-coeff@A.T)**2,axis=0)))
