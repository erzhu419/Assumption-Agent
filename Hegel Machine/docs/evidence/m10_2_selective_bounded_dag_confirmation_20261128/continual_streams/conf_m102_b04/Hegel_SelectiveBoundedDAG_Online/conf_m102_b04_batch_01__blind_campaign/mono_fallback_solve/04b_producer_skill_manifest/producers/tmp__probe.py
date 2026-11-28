import pickle,numpy as np
from itertools import combinations_with_replacement as C
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def basis(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,i],1) for k in range(1,d+1) for i in C(range(x.shape[1]),k)])
x=s['train']['x'];y=s['train']['y'];z=s['train']['traj'];U,S,V=np.linalg.svd(z,full_matrices=False);a=z@V[:2].T
print('times',s['eval_times'],'time basis',V[:2]);np.set_printoptions(precision=4,suppress=True)
for name in ('panel_a','panel_b','panel_c'):
 p=s[name];u=p['u'];print(name, 'ranges',u.min(),u.max());
 for key,v in p.items():
  if key=='u':continue
  for d in (1,2):
   f=basis(u,d); b=np.linalg.lstsq(f,v,rcond=None)[0]; e=v-f@b; h=np.sum(np.linalg.qr(f)[0]**2,axis=1);print(key,d,'rms/loo',np.std(e),np.std(e/(1-h[:,None])),'b',b.T if d==1 else '')
   if name=='panel_a' and d==1:
    for _ in range(8):
     k=np.argsort(np.linalg.norm(v-f@b,axis=1))[:41];b=np.linalg.lstsq(f[k],v[k],rcond=None)[0]
    print('robust',b,'residual norms',np.sort(np.linalg.norm(v-f@b,axis=1)))
    np.save('/tmp/cal.npy',b)
print('y and amplitude: degree profile')
for d in (1,2,3,4):
 f=basis(x,d)
 for reg in (0.0001,.01,.1):
  b=np.linalg.solve(f[:52].T@f[:52]+reg*np.eye(f.shape[1]),f[:52].T@np.c_[y,a][:52]);e=f@b-np.c_[y,a];print(d,reg,np.mean(abs(e[:52]),0),np.mean(abs(e[52:]),0))
 if d<=3:print('full OLS coefficients',np.linalg.lstsq(f,np.c_[y,a],rcond=None)[0].T)
