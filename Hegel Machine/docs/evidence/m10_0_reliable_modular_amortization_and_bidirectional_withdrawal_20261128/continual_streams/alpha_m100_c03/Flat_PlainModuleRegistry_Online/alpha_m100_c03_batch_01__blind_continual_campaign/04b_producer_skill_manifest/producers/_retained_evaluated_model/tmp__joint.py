import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from probe import *
f=basis(tau); z=h[:,None]-amp0/theta[:2]; one=np.ones(len(h)); j=h-p[:,-1]; base=np.c_[h[:,None]*theta[:2],np.zeros(len(h))]; Y=y-base@f.T

def fitlaw(cols, mask=None):
 if mask is None:mask=np.ones(len(h),bool)
 blocks=[v[:,None,:]*f[None,:,k,None] for k,v in enumerate(cols)]; A=np.concatenate(blocks,axis=2); m=A[mask].reshape(-1,A.shape[-1]);target=Y[mask].ravel(); w=np.linalg.lstsq(m,target,rcond=None)[0];w=least_squares(lambda w:m@w-target,w,loss='soft_l1',f_scale=.003,max_nfev=60).x
 return A@w+base@f.T,np.split(w,np.cumsum([x.shape[1] for x in cols])[:-1])
def mat(*cols):return np.column_stack(cols)
C=[mat(one,z[:,0],dr),mat(one,z[:,1],dr),mat(one,h,dr)]
yp,w=fitlaw(C); print('coeffs',w); E=y-yp; co=E@np.linalg.pinv(f).T; print('joint residual coeff',np.c_[co,dr,h,j,di['lab_id'],np.isnan(di['coupon_features_raw']).sum(1)]); print('error by time',np.sqrt(np.mean(E**2,0)))
features={'h':h,'j':j,'z0':z[:,0],'z1':z[:,1],'dr':dr,'dr2':dr**2,'dr3':dr**3,'dr*h':dr*h,'dr*z1':dr*z[:,1],'dr*z0':dr*z[:,0], 'dr*j':dr*j,'z1^2':z[:,1]**2,'z0^2':z[:,0]**2,'h^2':h*h,'j^2':j*j,'j^3':j**3,'pmean':p.mean(1),'pstd':p.std(1),'penergy':(p*p)@dt,'z1sat':z[:,1]/(1+z[:,1]**2),'tanhdr':np.tanh(dr),'tanh2dr':np.tanh(2*dr),'labs':di['lab_id'],'strata':di['stratum']}
features.update({f'p{k}':p[:,k] for k in range(p.shape[1])}); features.update({f'x{k}':Xd[:,k] for k in range(4)}); features.update({f'pd{k}':p[:,k+1]-p[:,k] for k in range(p.shape[1]-1)})
for k in range(3):
 print('branch',k,sorted([(abs(np.corrcoef(v,co[:,k])[0,1]),name) for name,v in features.items()],reverse=True)[:12])
print('aux train and test first 6',tr['aux_chamber_telemetry'][:6],te['aux_chamber_telemetry'][:6]);print('diag fields',di.keys()); print('refs by lab and split')
for lab in np.unique(tr['lab_id']):
 print(lab,[(q['ref_coupons'][q['lab_id']==lab].mean(),np.median(q['ref_coupons'][q['lab_id']==lab].std(0))) for q in (tr,te,di)])
