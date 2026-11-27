import numpy as np, pickle,json
from scipy.optimize import least_squares
z=np.load('/tmp/chamber_probe.npz'); p,h,x,dt,t,tau,th,Y,v=[z[k] for k in ('past','hold','features','dt','t','taus','theta','curve','rate')]; n=len(h); after=np.cumsum(dt[::-1])[::-1]-dt
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=it['dataset']; q=it['campaign_diagnostics']; tr=d['train']; te=d['test']; print('ROOT',list(it),list(d))
for e in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 print('LOG',[(k,(np.array(v).shape,np.round(np.array(v,dtype=float).mean(0),4).tolist()) if isinstance(v,list) else v) for k,v in e.items()])
F=lambda a: -np.expm1(-t[:,None]/a)
fit=least_squares(lambda b:(np.c_[F(np.exp(b)),t]@np.linalg.lstsq(np.c_[F(np.exp(b)),t],Y.T,rcond=None)[0]-Y.T).ravel(),np.log(tau)); print('diag free shape',np.exp(fit.x),np.std(fit.fun));
a=np.exp(fit.x); D=np.c_[F(a),t]; coeff=np.linalg.lstsq(D,Y.T,rcond=None)[0]; print('rate derivative fit',np.round(np.c_[v,(np.r_[1/a,1]@coeff),v-np.r_[1/a,1]@coeff],4));
for name,ref,lab in [('train/test',np.vstack((tr['ref_coupons'],te['ref_coupons'])),np.r_[tr['lab_id'],te['lab_id']]),('diag',q['ref_coupons'],q['lab_id'])]:
 xx=q['coupon_features_raw'].copy(); print(name)
 for l in np.unique(lab):
  R=ref[lab==l]; c,s=np.median(R.mean(0)),np.median(R.std(0)); print(l,c,s); xx[q['lab_id']==l]=(xx[q['lab_id']==l]-c)/s
 xx=np.where(np.isfinite(xx),xx,np.where(np.isnan(q['coupon_features_raw']),x,0)); r=th[2]+xx@th[3:]
 for a in [tau,np.exp(fit.x)]:
  f=F(a); m=p@(-np.expm1(-dt[:,None]/a)*np.exp(-after[:,None]/a)); base=h[:,None]*(f@th[:2]); rem=Y-base; rate=v-h@(np.ones((n,1))) if False else v-h*(th[:2]/a).sum()-r
  def cv(A,b):
   B=np.linalg.lstsq(A,b,rcond=None)[0]; Q=np.linalg.qr(A,mode='reduced')[0]; R=(b-A@B)/(1-(Q*Q).sum(1)).reshape((-1,)+(1,)*(b.ndim-1)); return np.round([np.sqrt(np.mean(R*R)),np.max(abs(R))],5),B
  for tag,A in [('intercept,r',np.c_[np.ones(n),r]),('r',r[:,None]),('r,m',np.c_[r,m]),('r,m2',np.c_[r,m[:,1]]),('r,hold',np.c_[r,h])]:
   met,B=cv(A,rem); print(np.round(a,4),tag,met,'latent',np.round(B@np.linalg.pinv(np.c_[f,t].T),4))
  print('rate no memory',np.round([np.std(rate),np.max(abs(rate)),np.mean(rate)],5)); print('rate with slow gap',cv(np.c_[np.ones(n),r,m],rate))
