import pickle,numpy as np
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; R=C['bridge_raw']; L=C['true_channel_loads']
PAIRS=((0,1),(0,3),(1,2),(2,3))
def basis(l):
 return np.column_stack([np.ones(len(l)),l,*[l[:,i]*l[:,j] for i,j in PAIRS],l**3])
def fit(l,r):
 a=basis(l); rng=np.random.default_rng(443); best=(np.inf,None)
 # Rank-one sweep excursions affect a minority of measurements.
 for j in range(350):
  ids=rng.choice(len(a),min(18,len(a)),replace=False); b=np.linalg.lstsq(a[ids],r[ids],rcond=None)[0]
  if np.linalg.matrix_rank(a[ids])<a.shape[1]: continue
  for _ in range(3):
   errs=np.linalg.norm(a@b-r,axis=1); keep=np.argsort(errs)[:int(.7*len(a))]; b=np.linalg.lstsq(a[keep],r[keep],rcond=None)[0]
  loss=np.mean(np.sort(np.linalg.norm(a@b-r,axis=1))[:int(.7*len(a))]**2)
  if loss<best[0]: best=loss,b
 b=best[1]; err=np.linalg.norm(a@b-r,axis=1); clean=err < max(0.00015,5*np.median(err)); return np.linalg.lstsq(a[clean],r[clean],rcond=None)[0]
def inv(r,b):
 shape=r.shape; r=r.reshape(-1,4); l=(r-b[0])@np.linalg.inv(b[1:5]); l=np.clip(l,-1.7,1.7)
 for k in range(25):
  e=basis(l)@b-r; jac=np.broadcast_to(b[1:5].T,(len(l),4,4)).copy(); jac+=b[9:].T[None,:,:]*3*l[:,None,:]**2
  for idx,(i,j) in enumerate(PAIRS):
   jac[:,:,i]+=l[:,j,None]*b[5+idx]; jac[:,:,j]+=l[:,i,None]*b[5+idx]
  step=np.linalg.solve(jac,e[...,None])[...,0]; l-=step/np.maximum(1,np.max(abs(step),axis=1,keepdims=True)/.4)
 return l.reshape(shape)
if __name__=='__main__':
 b=fit(L,R); e=R-basis(L)@b; q=np.linalg.norm(e,axis=1); print('full residual quantiles',np.quantile(q,[0,.25,.5,.65,.7,.75,.8,.9,.95,1])); print('contaminated rows',np.where(q>.001)[0]); print('coefficients',b.round(5)); print('linear singular values',np.linalg.svd(b[1:5],compute_uv=False))
 for fold,te in enumerate(np.array_split(np.random.default_rng(55).permutation(len(L)),5)):
  tr=np.setdiff1d(np.arange(len(L)),te); bf=fit(L[tr],R[tr]); qc=np.linalg.norm(R[te]-basis(L[te])@bf,axis=1); clean=q[te]<.001; pred=inv(R[te][clean],bf); de=(pred-L[te][clean]).sum(1); print('fold',fold,'clean',sum(clean),'forward max',max(qc[clean]),'total inverse RMSE/max',np.sqrt(np.mean(de**2)),max(abs(de)))
 for name in ('train','test'):
  z=D[name]; raw=np.concatenate((z['past_channel_raw'].reshape(-1,4),z['hold_channel_raw'])); l=inv(raw,b); err=np.linalg.norm(basis(l)@b-raw,axis=1); print(name,'inverse residual',np.quantile(err,[.5,.95,1]),'load quantiles',np.quantile(l,[0,.05,.95,1],axis=0))
