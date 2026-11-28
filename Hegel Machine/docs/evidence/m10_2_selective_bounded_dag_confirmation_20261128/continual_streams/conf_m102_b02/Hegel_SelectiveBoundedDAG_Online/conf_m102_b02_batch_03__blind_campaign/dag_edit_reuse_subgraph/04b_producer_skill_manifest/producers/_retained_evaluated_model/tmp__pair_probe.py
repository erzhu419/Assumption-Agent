import numpy as np,pickle
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(s['panel_a'][k] for k in ('u','v')); m=np.linalg.norm(v-u,axis=1)<4; u,v=u[m],v[m]; A=np.c_[np.ones(len(u)),u]; p=u*np.roll(u,-1,axis=1)
print('telemetry',s['telemetry_vec']); np.set_printoptions(precision=6,suppress=True)
for name,q in [('product',p),('tanh_product',np.tanh(1.25*p)),('product_denom',p/(1+.22*p*p)),('sum_denom',p/(1+.22*(u+np.roll(u,-1,axis=1))**2)),('sin_product',np.sin(1.25*p)),('tanh_neighbor',u*np.tanh(1.25*np.roll(u,-1,axis=1))),('product_cubic',np.c_[p,p**3]),('product_quadratic',np.c_[p,p**2])]:
 F=np.c_[A,q]; b=np.linalg.lstsq(F,v,rcond=None)[0]; r=v-F@b; h=np.sum(F*np.linalg.pinv(F).T,axis=1); print(name,np.sqrt(np.mean(r*r,axis=0)),np.sqrt(np.mean((r/(1-h[:,None]))**2,axis=0))); print(b[5:])
from itertools import combinations_with_replacement
terms=list(combinations_with_replacement(range(4),2)); F=np.c_[A,*[np.prod(u[:,t],axis=1) for t in terms]]; b=np.linalg.lstsq(F,v,rcond=None)[0]; print(list(zip(terms,b[5:]))); print('affine',b[:5]); print('max',np.max(abs(F@b-v),axis=0))
