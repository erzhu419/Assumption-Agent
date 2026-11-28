exec(open('/tmp/chamber_grammar.py').read().split("print('aux summaries'")[0])
xx=X['diag only']; v=np.c_[np.ones(len(h)),xx]@th[2:]; complete=np.isfinite(raw).all(1)
for mask in [complete,~complete,np.ones(len(h),bool)]:
 b=np.linalg.lstsq(v[mask,None]*f[None,:,:] if False else (v[mask,None,None]*f[None,:,:]).reshape(-1,3), R[mask].ravel(), rcond=None)[0]; print('complete/imputed/all',mask.sum(),b, 'RMS',np.sqrt(np.mean((R[mask]-v[mask,None]*(f@b))**2,axis=0)))
for b in ([.18,2.25,.24],[.18,2.2,.24]):
 e=R-v[:,None]*(f@b); print('rounded',b,'curve/rate max',np.max(abs(e[:,:-1])),np.max(abs(e[:,-1])),'complete max',np.max(abs(e[complete])))
print('test feature/drift shift by lab: mean joint, deployment only, difference RMS');
vt0=np.c_[np.ones(len(te['lab_id'])),x[n:]]@th[2:]; vt1=np.c_[np.ones(len(te['lab_id'])),a._features(te,empty)]@th[2:];
for l in np.unique(te['lab_id']):
 m=te['lab_id']==l; print(l,vt0[m].mean(),vt1[m].mean(),np.sqrt(np.mean((vt0[m]-vt1[m])**2)))
