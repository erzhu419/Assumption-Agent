exec(open('/tmp/probe11.py').read().split("print('state'")[0])
U=s['panel_a']['u']; V=s['panel_a']['v']; delta=np.linalg.norm(V-U,axis=1); f=a['_design'](U,35); print('cal rows sorted [idx, delta, err]'); print(np.c_[np.arange(len(U)),delta,np.max(abs(f@coef-V),axis=1)][np.argsort(delta)])
for j in range(4):
 k=(j+2)%4; x=U[:,j]; w=U[:,k]; M=np.c_[o[:len(U)],x,w,x*x,x*w,w*w,x**3,x*x*w,x*w*w,w**3]; cand=np.random.default_rng(14).integers(0,len(U),(3200,12)); mc=np.linalg.pinv(M[cand])@V[cand,j,None]; err=abs(M@mc[:,:,0].T-V[:,j,None]); k0=np.argmin(np.median(err,axis=0)); keep=err[:,k0]<max(.002,6*np.median(err[:,k0])); fit=ls(M[keep],V[keep,j]); print('pair',j,'inliers',keep.sum(),'coef',fit,'err',np.median(abs(M@fit-V[:,j])), 'excluded',np.where(~keep)[0]);
print('state residual per row:',np.c_[q,np.mean(abs(state@ls(state,z)-z),axis=1)][np.argsort(np.mean(abs(state@ls(state,z)-z),axis=1))[-12:]])
