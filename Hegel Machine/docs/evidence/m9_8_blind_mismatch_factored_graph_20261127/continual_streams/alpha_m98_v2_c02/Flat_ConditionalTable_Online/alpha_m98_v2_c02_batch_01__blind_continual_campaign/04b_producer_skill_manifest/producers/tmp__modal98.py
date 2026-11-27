exec(open('/tmp/law98.py').read().split('for mode in range')[0])
lab=a['lab_id']; n=len(h); I=np.arange(n); hs=s@W; rev=s[:,::-1]@W; print('archive assets summary:'); import json
J=json.load(open('/holdout_task/public_asset_index.json'));print([(k,str(v)[:300]) for k,v in J.items()])
for key in ['lab_id','stratum','coupon_features_raw','ref_coupons']:
 print(key,'diagnostic first row/shape',np.shape(a[key]),a[key][0]);

def tryfit(idx,kind):
 def calc(p,ret=False):
  fast,slow,dr=p[:3]; H0,H1,HD=map(expb,[fast,slow,dr]);
  if kind=='free': G=0; xs=[(np.c_[h,s],H0),(np.c_[h,s],H1),(q[:,None],HD)]
  else:
   G=h[:,None]*(b[0]*H0+b[1]*H1); baseH=b[0]*H0+b[1]*H1
   X={'zero':[], 'chron':[(-hs[:,0],b[0]*H0),(-hs[:,1],b[1]*H1)],'rev':[(-rev[:,0],b[0]*H0),(-rev[:,1],b[1]*H1)],'linear':[(s,baseH)],'abs':[(abs(s),baseH)],'revu':[(s[:,:2],baseH)],'revu1':[(s[:,0],baseH)],'revu1off':[(np.c_[s[:,0],np.ones(n)],baseH)],'revu1q':[(s[:,0],HD)],'revu1qoff':[(np.c_[s[:,0],np.ones(n)],HD)],'revu1qhold':[(np.c_[s[:,0],np.ones(n),h],HD)],'labs':[(np.eye(4)[lab],HD)],'strain0':[(a['strain_0'],HD)]}[kind]
   xs=[(q[:,None],HD)]+X
  Z=np.concatenate([np.reshape(x,(n,-1))[:,None,:]*v[None,:,None] for x,v in xs],2)
  C=np.linalg.lstsq(Z[idx].reshape(-1,Z.shape[-1]),(Y-G)[idx].ravel(),rcond=None)[0]; E=G+Z@C-Y
  return (E,C) if ret else E[idx].ravel()
 p=least_squares(calc,[u[0],u[1],2.1],bounds=([.45,1.2,.4],[.8,12,8]),max_nfev=100).x;return (*calc(p,True),p)
for kind in ['zero','chron','rev','linear','abs','revu','revu1','revu1off','revu1q','revu1qoff','revu1qhold','labs','strain0','free']:
 E,C,p=tryfit(I,kind); loo=np.stack([tryfit(I[I!=i],kind)[0][i] for i in I]); lolo=np.stack([tryfit(I[lab!=lab[i]],kind)[0][i] for i in I]); print(kind,p,C,'fit',rm(E)/[1,1,1,.7],'loo',rm(loo)/[1,1,1,.7],'lolo',rm(lolo)/[1,1,1,.7]);
