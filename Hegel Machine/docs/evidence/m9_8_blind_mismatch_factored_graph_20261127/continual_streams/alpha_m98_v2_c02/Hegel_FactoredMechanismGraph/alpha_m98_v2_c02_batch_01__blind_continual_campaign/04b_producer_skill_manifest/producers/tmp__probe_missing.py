exec(open('/tmp/probe_law.py').read().split('for kind in')[0])
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
obs=np.isfinite(D['coupon_features_raw']).all(1)
for key,X in [('diag',xd),('joint',m['_standardize'](tr,D)[1])]:
 for fill in ('existing','zero'):
  X=X.copy()
  if fill=='zero': X[np.isnan(D['coupon_features_raw'])]=0
  g=(beta[2]+X@beta[3:])[:,None]
  for kind in ('relative','scaled_tau','free_hold','free_tau'):
   p,e=calc(g,kind,obs)
   print(key,fill,kind,np.round(p,5),'clean/missing strain,rate',[(round(float(np.sqrt(np.mean(e[k,:-1]**2))),5),round(float(np.sqrt(np.mean(e[k,-1]**2))),5)) for k in (obs,~obs)])
  print('missing drift',g[~obs].ravel())
G=rd[:,None]; p,e=calc(G,'relative',obs); B=modes(tau); u=p@B; delta=y-(h[:,None]*beta[:2])@B; inferred=delta@u/(u@u); print('missing raw/implied drift',np.c_[D['lab_id'][~obs],rd[~obs],inferred[~obs]]); print('reference scale by lab and panel')
for name,s in [('train',tr),('test',te),('diag',D)]:
 for lab in np.unique(s['lab_id']):
  k=s['lab_id']==lab;r=s['ref_coupons'][k];print(name,lab,k.sum(),np.round([np.nanmedian(np.nanmean(r,0)),np.nanmedian(np.nanstd(r,0))],4),'missing',np.isnan(s['coupon_features_raw'][k]).sum())
print('inferred complete-case drift error',np.round((inferred-rd)[obs],4))
