import numpy as np,pickle,itertools as it
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_c']['u']; z=np.c_[s['panel_c']['v1'],s['panel_c']['v2']]; np.set_printoptions(precision=4,suppress=True,linewidth=150)
a=np.c_[np.ones(len(u)),u]; print('affine',np.linalg.lstsq(a,z,rcond=None)[0]);
for j in range(8):
 b=z[:,j]; pool=[]
 def add(name,val):pool.append((name,val))
 for i,k in it.combinations_with_replacement(range(4),2):add(f'u{i}u{k}',u[:,i]*u[:,k])
 for i in range(4):
  for w in np.arange(.5,5.25,.25):
   for name,fn in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh)]:add(f'{name} {w} u{i}',fn(w*u[:,i]))
 for i,k in it.combinations(range(4),2):
  for sign in [-1,1]:
   for w in [.5,1,1.5,2,2.5,3]:
    for name,fn in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh)]:add(f'{name} {w} (u{i}+{sign}u{k})',fn(w*(u[:,i]+sign*u[:,k])))
 F=np.column_stack([v for k,v in pool]); names=[]; A=a.copy()
 print('col',j,end=' ')
 for _ in range(4):
  R=F-A@np.linalg.lstsq(A,F,rcond=None)[0]; e=b-A@np.linalg.lstsq(A,b,rcond=None)[0]; k=np.argmax(abs(e@R)/np.maximum(np.linalg.norm(R,axis=0),1e-10)); A=np.c_[A,F[:,k]]; h=np.linalg.pinv(A); err=(b-A@(h@b))/(1-np.einsum('ij,ji->i',A,h)); print((pool[k][0],round(np.mean(abs(err)),6),round((h@b)[-1],4)),end=' ')
 print()
