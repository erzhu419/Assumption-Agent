exec(open('/tmp/explore06.py').read().split('for nm,F')[0]); from itertools import combinations_with_replacement
print('keys',s.keys()); print('time',s['eval_times']); print('panels',[(k,{j:(v.shape,np.quantile(v,[0,.1,.5,.9,1])) for j,v in s[k].items() if isinstance(v,np.ndarray)}) for k in ['panel_a','panel_b']]); u,v=b['u'],b['v']; print('panel B samples',u[:5],v[:5]); print('B affine per column quantiles',np.quantile(abs(a(u)@B-v),[0,.25,.5,.75,.9,1],axis=0));
def pol(x,d): return np.column_stack([np.prod(x[:,j],axis=1) for p in range(d+1) for j in combinations_with_replacement(range(x.shape[1]),p)])
for d in [1,2,3,4]:
 F=pol(u,d); coef=fit(F,v); e=F@coef-v; H=np.sum(np.linalg.svd(F,full_matrices=False)[0]**2,axis=1); print('panel degree',d,F.shape,'err med and rms',np.median(abs(e),axis=0),np.sqrt(np.mean(e**2,axis=0)),'loo med',np.median(abs(e)/np.maximum(1e-12,1-H[:,None]),axis=0));
 if d<=3:
  for k in range(4):
   ix=np.argsort(abs(coef[:,k]))[-12:][::-1]; names=[j for p in range(d+1) for j in combinations_with_replacement(range(4),p)]; print(k,[(names[j],round(coef[j,k],5)) for j in ix]);
print('telemetry',s['telemetry_vec']); print('ranges z',np.quantile(z,[0,.1,.5,.9,1],axis=0));
