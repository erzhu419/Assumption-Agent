import numpy as np,pickle,itertools as it
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=3,suppress=True)
for k,p in s.items():
 if not isinstance(p,dict):continue
 print(k,{a:np.shape(b) for a,b in p.items()});
 if not k.startswith('panel'):continue
 u=p['u']; d=np.c_[np.ones(len(u)),u]; print('u range',u.min(0),u.max(0))
 for name,v in p.items():
  if name=='u':continue
  keep=np.ones(len(u),bool)
  if k=='panel_a':keep=np.linalg.norm(v-d@np.linalg.lstsq(d,v,rcond=None)[0],axis=1)<3
  U=u[keep];V=v[keep]; D=d[keep];
  fs={'linear':D,'quad':np.c_[D,*[U[:,i]*U[:,j] for i,j in it.combinations_with_replacement(range(4),2)]],'sat':np.c_[D,-np.expm1(-.2962*U**2)],'sin':np.c_[D,np.sin(U)],'tanh':np.c_[D,np.tanh(U)],'trig':np.c_[D,np.sin(U),np.cos(U)],'tanhquad':np.c_[np.ones(len(U)),np.tanh(U),*[np.tanh(U[:,i])*np.tanh(U[:,j]) for i,j in it.combinations_with_replacement(range(4),2)]]}
  for nm,f in fs.items():
   b=np.linalg.lstsq(f,V,rcond=None)[0];H=np.linalg.svd(f,full_matrices=False)[0][:,:np.linalg.matrix_rank(f)];e=V-f@b;print(name,nm,'rms',np.sqrt(np.mean(e**2,0)),'loo',np.sqrt(np.mean((e/np.maximum(1-(H*H).sum(1),1e-8)[:,None])**2,0)))
  if k=='panel_a':
   for j in range(4):
    z=U[:,j];w=U[:,(j+1)%4];v=V[:,j]; print('A',j,'sorted',np.c_[z,w,v-D@np.linalg.lstsq(D,v,rcond=None)[0]][np.argsort(z)][::4])
