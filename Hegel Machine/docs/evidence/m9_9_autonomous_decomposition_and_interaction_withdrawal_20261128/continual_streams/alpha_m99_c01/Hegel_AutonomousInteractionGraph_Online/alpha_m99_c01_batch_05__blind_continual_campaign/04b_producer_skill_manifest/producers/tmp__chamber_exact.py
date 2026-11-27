from chamber_cv import *
miss=np.isnan(q['coupon_features_raw']).any(1)
z=feat('self'); d=beta[2]+z@beta[3:]; h=hq; labs=q['lab_id']

def fit(rows,kind,free,robust=False):
 def profile(u):
  ts=tau.copy()
  if free==1: ts[1]*=np.exp(u[0])
  if free==2: ts*=np.exp(u)
  K=modes(ts); base=0
  if kind!='free': base=h[:,None]*beta[0]*K[None,:,0]
  if kind=='anchor2': base=base+h[:,None]*beta[1]*K[None,:,1]
  B=[np.c_[h,d],np.c_[h,d],d[:,None]] if kind=='free' else [d[:,None],(d[:,None] if kind=='anchor2' else np.c_[h,d]),d[:,None]]
  A=np.concatenate([v[:,None,:]*K[None,:,j,None] for j,v in enumerate(B)],axis=2)
  a=A[rows].reshape(-1,A.shape[-1]); yy=(Y-base)[rows].ravel(); w=np.ones(len(yy))
  for _ in range(7 if robust else 1):
   c=np.linalg.lstsq(a*w[:,None],yy*w,rcond=None)[0]; e=a@c-yy; w=(1+(e/.004)**2)**(-.25)
  return e,A@c+base,c,ts
 u=least_squares(lambda v:profile(v)[0],np.zeros(free),bounds=(-.8,.8),loss='soft_l1',f_scale=.004,diff_step=1e-4).x if free else []
 return profile(u)[1:]

def rm(e): return np.sqrt([np.mean(e[:,:-1]**2),np.mean(e[:,-1]**2)])
for kind in ['free','anchor1','anchor2']:
 for clk in range(3):
  pred=np.zeros_like(Y); worst=[]
  for l in np.unique(labs):
   v=labs==l; pred[v]=fit(~v&~miss,kind,clk,True)[0][v]; worst.append(rm((pred-Y)[v&~miss]))
  p,c,ts=fit(~miss,kind,clk,True); print(kind,clk,'worst complete',np.round(np.max(worst,axis=0),6),'oof missing',np.round(rm((pred-Y)[miss]),6),'in complete',np.round(rm((p-Y)[~miss]),6),'ratio',np.round(ts/tau,6),'c',np.round(c,6))
  if kind=='anchor2' and clk==1:
   k=modes(ts); drshape=k@c; base=h[:,None]*(k[:,:2]@beta[:2]); infer=(Y-base)@drshape/(drshape@drshape); print('latent drift vs chosen',np.round(np.c_[d,infer][miss],5)); print('imputed entries of features',z[miss], 'missing coordinates',np.argwhere(np.isnan(q['coupon_features_raw'])))
