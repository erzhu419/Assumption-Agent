import itertools as it,pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_a']['u'];v=s['panel_a']['v']; np.set_printoptions(precision=5,suppress=True,linewidth=145)
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
keep=np.linalg.norm(v,axis=1)<4
print('excluded',np.where(~keep)[0]);print('excluded values',v[~keep]); print('retained',sum(keep))
U=u[keep]; V=v[keep]
a=np.c_[np.ones(len(U)),U]; q=ls(a,V); r=V-a@q;print('joint affine',q,'rms',np.sqrt(np.mean(r*r,axis=0)))
for j in range(4):
 print('output',j)
 for i in range(4):
  for k in range(i,4):
   t=U[:,i]*U[:,k]; d=t-a@ls(a,t); co=d@r[:,j]/(d@d); rms=np.std(r[:,j]-co*d)
   if rms<.97*np.std(r[:,j]): print('pair',i,k,co,'rms',rms)
 for p in (2,3,4):
  powers=[inds for k in range(2,p+1) for inds in it.combinations_with_replacement(range(4),k)]
  d=np.c_[a,*[U[:,inds].prod(1) for inds in powers]]; q=ls(d,V[:,j]); res=d@q-V[:,j]; lev=np.linalg.norm(np.linalg.qr(d,mode='reduced')[0],axis=1)**2
  print('degree',p,'rms',np.std(res),'LOO MAE',np.mean(abs(res/np.maximum(1-lev,1e-12))),'strong terms',[(inds,round(float(z),5)) for inds,z in zip([(),*( (k,) for k in range(4)),*powers],q) if abs(z)>.024])
 print('correlations residual',np.corrcoef(np.c_[r[:,j],U**3,np.sin(1.25*U), U/(1+1.25*U**2)].T)[0,1:])
