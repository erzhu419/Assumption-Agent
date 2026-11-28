import pickle, runpy, numpy as np
p='/holdout_task/'
m=runpy.run_path(p+'node_00.py'); b=runpy.run_path(p+'_frozen_nodes/node_01.py')['run_node']; s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,z=(tr[k] for k in ('x','y','traj')); n=len(x)
q=m['_coordinates'](s,x); g=m['_learn'].__globals__; base=g['_kernel']; a=s['panel_a']; f=np.c_[np.ones(len(a['u'])),a['u']]; k=np.arange(len(f));
for _ in range(8):
    c=np.linalg.lstsq(f[k],a['v'][k],rcond=None)[0]; r=np.linalg.norm(f@c-a['v'],axis=1); k=np.argsort(r)[:int(.8*len(f))]
print('panel_a matrix:',np.round(c,4),'trimmed residual',np.median(r))
splits=[np.arange(20),np.arange(26,46),np.arange(n-20,n),np.argsort(np.sum(((x-x.mean(0))/x.std(0))**2,axis=1))[-12:]]
for label,qq in [('calibrated',q),('atan_coordinates',np.c_[q,np.arctan(2*q[:,:4])]),('raw_atan',np.c_[q[:,:4],np.arctan(2*q[:,:4])]),('calibrated_v1',q[:,:8])]:
    vals=np.empty((4,4,2))
    for i,va in enumerate(splits):
        it=np.setdiff1d(np.arange(n),va); idx=np.r_[it,va]; nt=len(it); sq=m['_dist'](qq[idx]); yt=y[it]; zt=z[it]; aa=zt@np.linalg.svd(zt,full_matrices=False)[2][0]; gy=m['_gram'](sq,m['_learn'](sq[:nt,:nt],yt)); ga=m['_gram'](sq,m['_learn'](sq[:nt,:nt],aa)); data={'train':{'x':x[it],'y':yt,'traj':zt}}; ff=m['_features']; af=ff(ga,nt,.014); hf=ff(gy,nt,.007)
        for j,w in enumerate([.25,.5,.75,1.]):
            out=b({'__input__':data,'node_00':{'raw':ff(w*ga+(1-w)*gy,nt,.014),'aligned':af,'hybrid':hf}}); vals[j,i]=[np.mean(abs(out['y_hat']-y[va])),np.mean(abs(out['state_pred']-z[va]))]
    for w,v in zip([.25,.5,.75,1.],vals): print(label,w,np.round(v,5).tolist(), 'worst',np.round(v.max(0),5).tolist())
