import numpy as np,pickle
from itertools import combinations_with_replacement as cwr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=5,suppress=True,linewidth=140)
u,v=[s['panel_a'][k] for k in ('u','v')]; mask=(np.max(abs(v),axis=1)<4); u,v=u[mask],v[mask]
def d(x,inds):return np.column_stack([np.prod(x[:,ij],axis=1) for ij in inds])
a=[()]+[(i,) for i in range(4)]; q=list(cwr(range(4),2)); c=list(cwr(range(4),3));
for name,I in [('aff',a),('q',a+q),('diag3',a+[(i,)*j for j in (2,3) for i in range(4)]),('qdiag3',a+q+[(i,)*3 for i in range(4)]),('full3',a+q+c),('diag5',a+[(i,)*j for j in range(2,6) for i in range(4)])]:
 D=d(u,I); coef=np.linalg.lstsq(D,v,rcond=None)[0]; r=v-D@coef; Q=np.linalg.qr(D,mode='reduced')[0]; print(name,'rmse',np.sqrt(np.mean(r*r,axis=0)),'loo',np.mean(abs(r/(1-(Q*Q).sum(1))[:,None]),axis=0));
 if name in ('qdiag3','diag5','full3'): print(list(zip(I,np.round(coef,4).tolist())))
print('affine residual by range',np.c_[u,v-d(u,a)@np.linalg.lstsq(d(u,a),v,rcond=None)[0]][:12])
