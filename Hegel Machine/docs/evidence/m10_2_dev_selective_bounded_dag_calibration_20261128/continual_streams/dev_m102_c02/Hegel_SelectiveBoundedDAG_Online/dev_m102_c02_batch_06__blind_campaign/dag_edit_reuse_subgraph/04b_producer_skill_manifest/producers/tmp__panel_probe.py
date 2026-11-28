import pickle,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True,linewidth=160)
for p in ['panel_a','panel_b','panel_c']:
 a=s[p]; u=a['u']; print('\n',p,'u range',u.min(0),u.max(0));
 for k in a:
  if k=='u': continue
  v=a[k]; d=np.c_[np.ones(len(u)),u]; q=np.column_stack([np.prod(u[:,j],axis=1) for j in combinations_with_replacement(range(4),2)]); w=np.linalg.lstsq(d,v,rcond=None)[0]; r=v-d@w; H=np.linalg.pinv(d); rq=v-np.c_[d,q]@np.linalg.lstsq(np.c_[d,q],v,rcond=None)[0]; print(k,'range',v.min(0),v.max(0),'affine coeff\n',w,'aff RMS',np.mean(r*r,0)**.5,'quadratic rms',np.mean(rq*rq,0)**.5); print('ordered abs affine residual\n',np.sort(np.abs(r),axis=0)[[0,5,10,15,21,-3,-2,-1]]); print('corr quadratic\n', ((q-d@H@q).T@r)/np.linalg.norm(q-d@H@q,axis=0)[:,None]/np.linalg.norm(r,axis=0));
print('observed x train/test',*[np.quantile(s[p]['x'],[0,.05,.5,.95,1],axis=0) for p in ('train','test')]); print('trajectory singular values',np.linalg.svd(s['train']['traj'],compute_uv=False))
