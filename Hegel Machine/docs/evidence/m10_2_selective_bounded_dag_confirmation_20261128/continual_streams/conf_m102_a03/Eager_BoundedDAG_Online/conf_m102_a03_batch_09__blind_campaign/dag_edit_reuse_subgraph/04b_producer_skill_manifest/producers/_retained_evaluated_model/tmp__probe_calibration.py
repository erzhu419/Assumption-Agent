import numpy as np,pickle,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(suppress=True,precision=5,linewidth=150)
fit=lambda a,b: np.linalg.lstsq(a,b,rcond=None)[0]
for name in ['panel_a','panel_b','panel_c']:
 p=s[name];u=p['u']; print(name,'u range',u.min(0),u.max(0))
 for k,v in p.items():
  if k=='u':continue
  a=np.c_[np.ones(len(u)),u]; c=fit(a,v); e=v-a@c
  print(k,'affine coeff',c,'singular',np.linalg.svd(c[1:],compute_uv=False),'row err',np.sort(np.linalg.norm(e,axis=1)))
  for j in range(4):
   x=u[:,j]; y=v[:,j]; r=u[:,(j+1)%4]; l=u[:,(j-1)%4]; o=u[:,(j+2)%4]; d=np.c_[a,x*x,x**3,r*r,r**3,l*l,l**3,o*o,o**3,x*r,x*l,x*o,r*l,r*o,l*o,x*x*r,x*r*r,x*x*l,x*l*l,x*x*o,x*o*o,r*l*o]; cf=fit(d,y); er=y-d@cf
   if name=='panel_a': print(j,'cubic simple',fit(np.c_[np.ones(len(x)),x,x**3,r,l,o],y),'full',cf,'err rms/max',np.std(er),max(abs(er)))
 if name=='panel_a':
  for j in range(4):
   x=u[:,j];y=p['v'][:,j];i=np.argsort(x);print('ordered pairs',j,np.c_[x[i],y[i]])
print('traj sv',np.linalg.svd(s['train']['traj'],compute_uv=False))
