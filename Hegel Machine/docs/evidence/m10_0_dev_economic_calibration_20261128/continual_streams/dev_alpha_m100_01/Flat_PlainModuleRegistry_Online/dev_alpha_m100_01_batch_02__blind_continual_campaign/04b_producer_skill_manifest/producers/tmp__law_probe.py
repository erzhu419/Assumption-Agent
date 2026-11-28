from chamber_probe import *
ix=np.arange(len(dg['lab_id'])); labs=dg['lab_id']; xraw,x=diagfeat(); past,h=loads(dg); y=obs(dg); ta,th=fit(np.arange(n)); B=basis(ta)
z=np.zeros((len(h),2))
for i,v in enumerate(dt): z=np.exp(-v/ta)*z-past[:,i,None]*np.expm1(-v/ta)
g=h[:,None]-z; s=th[2]+x@th[3:]; baseline=np.c_[g*th[:2],s]; e=y-baseline@B.T
print('baseline worst horizon and rate RMSE',np.sqrt(np.mean(e**2,axis=0))[-2:]); print('delta coefficient correlations'); inputs=np.c_[h,z,g,s,s*s,s**3,x,past]; print(np.corrcoef(np.c_[e@np.linalg.pinv(B.T),inputs].T)[:3,3:].round(3)); print('free deltas vs inputs'); print(np.c_[g,s,e@np.linalg.pinv(B.T)].round(4))
one=np.ones(len(h)); col=lambda *args: np.column_stack(args); empty=np.empty((len(h),0)); groups=[ix==k for k in ix]+[labs==k for k in np.unique(labs)]
def evaluate(parts):
 A=np.concatenate([B[None,:,j,None]*p[:,None,:] for j,p in enumerate(parts)],axis=2); pred=[]; coeff=[]
 for m in [np.zeros(len(ix),bool)]+groups:
  a=A[~m].reshape((~m).sum()*len(B),-1); k=np.linalg.lstsq(a,e[~m].ravel(),rcond=None)[0]; pred.append((e-A@k)[m] if m.any() else e-A@k); coeff.append(k)
 def rm(v): return np.r_[np.sqrt(np.mean(v[:,:-1]**2)),np.sqrt(np.mean(v[:,-1]**2))]
 return np.round(np.r_[rm(pred[0]),rm(np.vstack(pred[1:len(ix)+1])),np.max([rm(v) for v in pred[len(ix)+1:]],0)],5),coeff[0].round(4)
options={'0':empty,'1':col(one),'s':col(one,s),'g0':col(g[:,0]),'g1':col(g[:,1]),'zg0':col(one,z[:,0],h),'zg1':col(one,z[:,1],h),'sg0':col(one,s,g[:,0]),'sg1':col(one,s,g[:,1]),'zsg0':col(one,s,z[:,0],h),'zsg1':col(one,s,z[:,1],h),'x':col(one,x),'s2':col(one,s,s*s),'s3':col(one,s,s*s,s**3),'h':col(one,h),'hs':col(one,h,s,h*s),'sat':col(one,s/(1+s*s)), 'gxy':col(one,s,g,s*g[:,0],s*g[:,1]), 'spt':col(one,s,past,h)}
for a,b in [('0','0'),('g0','g1'),('0','g1'),('g0','0'),('s','s'),('sg0','sg1'),('zsg0','zsg1'),('spt','spt')]:
 for c in ('0','1','s','s2','s3','x','hs','sat','gxy','spt'):
  met,k=evaluate([options[j] for j in (a,b,c)]); print(a,b,c,met,k) if met[4]<.08 or (a==b=='0') else None
print('standardization diagnostics-only difference'); dx=prior._features(dg,dg)[:len(ix)]; print(np.max(abs(dx-x),axis=0)); print('test missing',np.isnan(test['coupon_features_raw']).sum(0)); print('diagnostic missing',np.isnan(xraw).sum(0))
