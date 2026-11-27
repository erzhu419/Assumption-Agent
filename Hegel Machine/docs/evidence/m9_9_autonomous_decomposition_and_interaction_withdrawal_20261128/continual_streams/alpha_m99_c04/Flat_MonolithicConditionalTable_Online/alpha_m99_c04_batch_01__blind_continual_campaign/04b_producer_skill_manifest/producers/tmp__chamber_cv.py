import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 exec(open('/tmp/chamber_probe.py').read().split('for g,x')[0])
from itertools import combinations
g=dg; y=g['strain_curves_obs']-g['strain_0'][:,None]; v=g['instant_rate_obs']; labs=g['lab_id']; past=st[0]+g['past_channel_raw']@st[1:]; hold=st[0]+g['hold_channel_raw']@st[1:]; age=np.r_[np.cumsum(dt[::-1])[::-1][1:],0]; z=past@(-np.expm1(-dt[:,None]/tau)*np.exp(-age[:,None]/tau))
stds={}
for name,ref in [('diag',dg),('train+diag',None),('train+test',te),('train',tr)]:
 if name=='train+diag':
  stds[name]=xd; continue
 if name=='train+test':
  ref={k:np.concatenate([tr[k],te[k]]) for k in ('ref_coupons','lab_id')}
 x=g['coupon_features_raw'].copy()
 for l in np.unique(labs):
  b=ref['ref_coupons'][ref['lab_id']==l]; center=np.median(b.mean(0)); scale=np.median(b.std(0)); x[labs==l]=(x[labs==l]-center)/scale
 _,cell,ct=np.unique(np.c_[labs,g['stratum']],axis=0,return_inverse=True,return_counts=True); w=1/ct[cell,None]; mu=np.nansum(w*x,axis=0)/np.sum(w*np.isfinite(x),axis=0); stds[name]=np.where(np.isfinite(x),x,mu)
E=-np.expm1(-t[:,None]/tau); B=np.column_stack([E,t]); R=np.r_[1/tau,1.]; N=len(y); ys=np.c_[y,v]; weights=np.r_[np.ones(len(t)),.6];
for name,x in stds.items():
 drift=th[2]+x@th[3:]; F=np.c_[np.ones(N),x]; base=hold[:,None]*th[:2]; init=np.c_[base,np.zeros(N)]; off=init@np.vstack([B,R]).T
 print('\n',name,'nans',np.isnan(g['coupon_features_raw']).sum(axis=0)); print('fixed tau coeff on [hold,1,x0,x1,x2,x3,z0,z1] rows fast/slow/drift:',np.round(np.linalg.lstsq(np.c_[hold,F,z],np.linalg.lstsq(np.vstack([B,R])*weights[:,None],(ys*weights).T,rcond=None)[0].T,rcond=None)[0].T,4));
 for typ in ('drift_redistributed','drift_plus_intercept','drift_free_features','drift_history','drift_full_channels','features_interactions'):
  if typ=='drift_redistributed': A=[drift[:,None]]*3
  if typ=='drift_plus_intercept': A=[np.c_[np.ones(N),drift]]*3
  if typ=='drift_free_features': A=[F]*3
  if typ=='drift_history': A=[np.c_[F,z[:,j]] for j in range(2)]+[F]
  if typ=='drift_full_channels': A=[np.c_[F,hold,z]]*3
  if typ=='features_interactions': A=[np.c_[F,drift**2,hold*drift,z]]*3
  D=np.concatenate([A[j][:,None,:]*np.r_[B[:,j],R[j]][None,:,None] for j in range(3)],axis=2)
  err=[]
  for l in np.unique(labs):
   a=labs!=l; b=~a; p=np.linalg.lstsq((D[a]*weights[None,:,None]).reshape(-1,D.shape[-1]),((ys[a]-off[a])*weights).ravel(),rcond=None)[0]; e=ys[b]-off[b]-D[b]@p; err.append(np.sqrt(np.mean(e[:,:-1]**2)),) if False else err.append([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))]);
  p=np.linalg.lstsq((D*weights[None,:,None]).reshape(-1,D.shape[-1]),((ys-off)*weights).ravel(),rcond=None)[0]; print(typ,'worst',np.round(np.max(err,axis=0),5),'mean',np.round(np.mean(err,axis=0),5),'coef',np.round(p,4) if typ in ('drift_redistributed','drift_plus_intercept','drift_history') else '')
for k in ('train','test','diag'):
 q={'train':tr,'test':te,'diag':g}[k]; raw=q['hold_channel_raw']; p=q['past_channel_raw']; print(k,'total hold range',np.min(st[0]+raw@st[1:]),np.max(st[0]+raw@st[1:]),'history means',np.round((st[0]+p@st[1:]).mean(0),3),'history std',np.round((p@st[1:]).std(0),3));
