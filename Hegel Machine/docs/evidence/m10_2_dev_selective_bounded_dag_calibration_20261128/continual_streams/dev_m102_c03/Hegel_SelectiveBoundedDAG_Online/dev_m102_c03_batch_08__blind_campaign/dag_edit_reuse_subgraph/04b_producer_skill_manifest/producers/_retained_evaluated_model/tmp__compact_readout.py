import sys,pickle,numpy as np,itertools
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task/_frozen_nodes'); import node_00 as f
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; N=len(y); ff=f.run_node({'__input__':s}); H=ff['ht'][:N]; z=f._inverse(x,f._calibration(s['panel_a'])); j=np.roll(z,-1,axis=1); ks=list(itertools.combinations(range(4),3)); C=np.column_stack([*[(z[:,k]).prod(1) for k in ks],*(z*z*j).T,*(z*j*j).T]); A=np.c_[H,z,C]; Y=np.c_[y,T]; fit=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]; w=fit(A,Y); print('coeffs columns: scalar, first, middle, final\n',w[:,[0,1,4,8]].round(5)); print('cubic trajectory weights relative to scalar', (w[-12:,1:]/w[-12:,0:1]).round(3)); print('ranks',[(q,np.linalg.matrix_rank(a)) for q,a in [('base',H),('plus z',A[:,:15]),('full',A)]]); print('baseline residual along sqrt',np.linalg.svd(Y-H@fit(H,Y),full_matrices=False)[2][0].round(4));
def temporal(a,tr,y,use_y):
 g=fit(a,tr); sw=fit(a,y); zt=g.copy(); zt[-12:]=0
 obs=a@zt
 def q(r):return np.column_stack((-np.expm1(-t[:,None]*r),np.sqrt(t)))
 def res(r):
  qt=q(r); return (obs-(obs@np.linalg.pinv(qt.T))@qt.T).ravel()
 rates=least_squares(res,[.42,1.82],bounds=([.05,.8],[.8,5])).x
 qt=q(rates); B=fit(qt,tr.T).T; Q=fit(a,B)
 if use_y: Q[-12:,-1]=sw[-12:]*.55
 print('rates',rates.round(4),'trajectory coeffs on sqrt',Q[:,-1].round(4),'basis residual',np.abs(tr-B@qt.T).mean().round(5))
 return Q@qt.T
folds=[(np.arange(52),np.arange(52,N))]+[(np.setdiff1d(np.arange(N),v),v) for v in np.array_split(np.random.default_rng(127).permutation(N),6)];
for name,a in [('H+z+cubic',A),('H+cubic',np.c_[H,C]),('H+triple',np.c_[H,C[:,:4]]),('H+edge cubic',np.c_[H,C[:,4:]]),('H+z+cubic+temporal',A),('H+z+cubic+joint',A)]:
 E=[]
 for tr,te in folds:
  p=a[te]@fit(a[tr],Y[tr]);
  if name.endswith(('temporal','joint')): p[:,1:]=a[te]@temporal(a[tr],T[tr],y[tr],name.endswith('joint'))
  err=np.abs(p-Y[te]);E.append([err[:,0].mean(),err[:,1:].mean(),err[:,1:].mean(0).max()])
 print(name,'public',np.round(E[0],5),'worst',np.round(np.max(E,axis=0),5))
