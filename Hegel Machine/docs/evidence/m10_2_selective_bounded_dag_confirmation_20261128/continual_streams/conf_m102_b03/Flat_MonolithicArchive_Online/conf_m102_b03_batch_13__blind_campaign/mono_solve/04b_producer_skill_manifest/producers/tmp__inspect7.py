import numpy as np,pickle
from scipy.optimize import minimize_scalar,least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; D=lambda x:np.c_[np.ones(len(x)),x]; L=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]; np.set_printoptions(precision=4,suppress=True)
u,v=s['panel_a']['u'],s['panel_a']['v'];g=np.arange(len(u));
for _ in range(5):
 c=L(D(u[g]),v[g]);r=np.linalg.norm(v-D(u)@c,axis=1);g=np.argsort(r)[:39]
print('A robust coeff',c,'sorted residual',np.sort(r),sep='\n');print('A sample',np.c_[u,v][:18]);
for j in range(4):
 a=u[:,j]; i,h=np.triu_indices(len(u),1); sl=np.median((v[i,j]-v[h,j])/(a[i]-a[h])); print('diag',j,sl,np.sort(abs(v[:,j]-sl*a-np.median(v[:,j]-sl*a))))
u=s['panel_c']['u'];
for key,f in [('v1',lambda q,k:D(np.c_[q,q/(1+k*abs(q))])),('v2',lambda q,k:D(np.c_[q,q*np.roll(q,-1,axis=1),np.arctan(k*q)]))]:
 y=s['panel_c'][key]; obj=lambda k:np.mean((f(u,k)@L(f(u,k),y)-y)**2);res=minimize_scalar(obj,bounds=(.2,2),method='bounded');print(key,res.x,res.fun**.5,L(f(u,res.x),y),sep='\n')
t=s['eval_times']; T=s['train']['traj'][:52]; H=lambda r:-np.expm1(-t[:,None]*r); e=lambda r:(H(r)@L(H(r),T.T)-T.T).ravel();r=least_squares(e,[2.15,.4],bounds=([.8,.01],[5,1.2]),ftol=1e-12).x; a=L(H(r),T.T).T; print('rates',r,'rmse',np.mean(e(r)**2)**.5,'y from amplitudes',np.mean(abs(D(a)@L(D(a),s['train']['y'][:52])-s['train']['y'][:52])), 'y triple',np.c_[s['train']['y'][:12],a[:12]]);print('B',L(D(s['panel_b']['u']),s['panel_b']['v']))
