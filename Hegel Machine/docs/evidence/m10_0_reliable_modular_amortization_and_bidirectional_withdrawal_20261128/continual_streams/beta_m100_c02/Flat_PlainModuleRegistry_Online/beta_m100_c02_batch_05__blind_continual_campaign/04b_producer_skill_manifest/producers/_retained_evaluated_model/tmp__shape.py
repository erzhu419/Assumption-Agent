import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; clean=np.max(abs(y),axis=1)<5
b=c['single_bank_sweeps']; u=b['ref_signal']; v=b['raw_readout']; j=b['bank_id']; good=abs(v)<5
# Include the corresponding diagonal of each isolated rosette sweep.
i=np.flatnonzero(clean & ((x!=0).sum(1)==1)); k=np.argmax(abs(x[i]),axis=1)
u=np.r_[u[good],x[i,k]]; v=np.r_[v[good],y[i,k]]; j=np.r_[j[good],k]; B=np.eye(4)[j]
def rat(u,p):return (u+p[0]*u**2+p[1]*u**3)/(1+p[2]*u*u+p[3]*u**4)
models={
 'rat22':(lambda u,p:rat(u,[p[0],0,p[1],0]),[0,1],([-1,0],[1,8])),
 'rat32':(lambda u,p:rat(u,[p[0],p[1],p[2],0]),[0,.1,1],([-2,-2,0],[2,5,12])),
 'rat34':(rat,[0,.1,1,.1],([-2,-2,0,0],[2,5,12,12])),
 'sqrt_num':(lambda u,p:(u+p[0]*u*u+p[1]*u**3)/np.sqrt(1+p[2]*u*u+p[3]*u**4),[0,0,1,.1],([-2,-2,0,0],[2,5,12,12])),
 'tanh_rat':(lambda u,p:np.tanh(p[0]*u)+ (p[1]*u+p[2]*u*u+p[3]*u**3)/(1+p[4]*u*u),[1,.4,0,.04,1],([.1,-3,-2,-2,0],[5,3,2,2,12])),
 'tanh_sin':(lambda u,p:np.tanh(p[0]*u)+p[1]*np.sin(p[2]*u)+p[3]*np.sin(p[4]*u)**2,[1,.15,2,.015,1],([.1,-3,.1,-2,.1],[5,3,8,2,8])),
 'dual_tanh':(lambda u,p:np.tanh(p[0]*u)+p[1]*np.tanh(p[2]*u)+p[3]*np.tanh(p[4]*u)**2,[1,.15,2,.015,1],([.1,-3,.1,-2,.1],[5,3,8,2,8])),
 'tanh_cubic':(lambda u,p:np.tanh(p[0]*u)+p[1]*u+p[2]*u*u+p[3]*u**3,[1,.15,.015,.01],([.1,-3,-2,-2],[5,3,2,2])),
}
def fit(f,p,bd,tr):
 def resid(a):
  D=np.column_stack([B, B*f(u,a)[:,None]])[tr]; return D@np.linalg.lstsq(D,v[tr],rcond=None)[0]-v[tr]
 return least_squares(resid,p,bounds=bd,gtol=1e-13,max_nfev=400)
fold=np.random.default_rng(3).permutation(len(u))%5
for name,(f,p,bd) in models.items():
 sol=fit(f,p,bd,np.ones(len(u),bool)); err=[]
 for k in range(5):
  tr=fold!=k; a=fit(f,p,bd,tr).x; D=np.column_stack([B,B*f(u,a)[:,None]]); coeff=np.linalg.lstsq(D[tr],v[tr],rcond=None)[0]; err.append(np.sqrt(np.mean((D[~tr]@coeff-v[~tr])**2)))
 print(name,'full',np.std(sol.fun),'CV',max(err),'parameters',np.round(sol.x,7))
# Compare separate own-channel nonlinear response and cyclic opposite/neighbor cross terms.
for name in ('rat32','tanh_rat','tanh_sin','tanh_cubic'):
 f,p,bd=models[name]; a=fit(f,p,bd,np.arange(len(u))<good.sum()).x
 print('mixed',name)
 for mode in range(5):
  fold=np.random.default_rng(8).permutation(len(x))%5; es=[]
  for k in range(5):
   tr=clean&(fold!=k);te=clean&(fold==k); pred=[]
   for j0 in range(4):
    z=f(x,a); pair=[x[:,i]*x[:,j] for i,j in combinations(range(4),2)]; cross=[x[:,i]*z[:,j] for i,j in combinations(range(4),2)]; D=np.column_stack([np.ones(len(x)),z,x,x*x]+[[],pair,cross,pair+cross,pair+[x[:,i]*x[:,j]**2 for i in range(4) for j in range(4) if i!=j]][mode]); pred.append(D[te]@np.linalg.lstsq(D[tr],y[tr,j0],rcond=1e-10)[0]);
   es.append(np.sqrt(np.mean((np.array(pred).T-y[te])**2)))
  print(mode,round(np.mean(es),7),round(max(es),7))
