import pickle,numpy as np,itertools
from scipy.sparse import diags,kron,eye
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; nx,ny=s['nx'],s['ny']; idx=s['sensor_ij']@np.array([1,nx]); mu=s['mu_shifts']; pulses=s['reference_tracer_pulses']; B=s['source_modes'].reshape(len(s['source_modes']),-1,order='F').T@np.column_stack([p['w_pulse'] for p in pulses]); Y=np.column_stack([p['y_pulse_raw'] for p in pulses]); I=eye(nx*ny,format='csc')
wall=np.zeros((nx,ny,4)); wall[0,:,0]=wall[-1,:,1]=wall[:,0,2]=wall[:,-1,3]=1; W=wall.reshape(-1,4,order='F')
def one(k,v,h,km,vm,form):
 d=(k[:-1]+k[1:])/2 if km==0 else 2*k[:-1]*k[1:]/(k[:-1]+k[1:]); d=d/h**2
 if vm==0: f=(v[:-1]+v[1:])/2/h; p=np.maximum(f,0); m=np.maximum(-f,0)
 elif vm==1: p=np.maximum(v[:-1],0)/h; m=np.maximum(-v[1:],0)/h
 else: p=np.maximum(v[1:],0)/h; m=np.maximum(-v[:-1],0)/h
 if form==0: a=np.r_[d+p,0]+np.r_[0,d+m]
 else: a=np.r_[d+m,0]+np.r_[0,d+p]
 return diags([-d-p,a,-d-m],[-1,0,1],format='csc')
def op(km,vm,form,spacing):
 x=one(s['kappa_x'],s['vel_x'],s['Lx']/(nx-spacing),km,vm,form); y=one(s['kappa_y'],s['vel_y'],s['Ly']/(ny-spacing),km,vm,form)
 return (kron(eye(ny),x)+kron(y,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
def predict(A,a,b=B):
 D=A+diags(W@a); return np.concatenate([splu(D+m*I).solve(b)[idx] for m in mu])
def fit(A,train=None):
 if train is None: train=np.ones(Y.shape[0],bool)
 return least_squares(lambda a:(predict(A,a)-Y)[train].ravel(),np.full(4,5.),bounds=(0.,200.),ftol=1e-11,gtol=1e-11,xtol=1e-11)
if __name__=='__main__':
 results=[]
 for key in itertools.product(range(2),range(3),range(2),range(2)):
  A=op(*key); f=fit(A); results.append((np.sqrt(np.mean(f.fun**2)),key,f.x))
 for rm,key,a in sorted(results)[:12]: print('walls fit',key,'rmse',rm,'wall diag',a)
 rm,key,a=min(results); A=op(*key)
 for fold in range(4):
  mask=np.tile((np.arange(len(idx))//4)%4==fold,len(mu)); f=fit(A,~mask); print('rosette jackknife',fold,np.sqrt(np.mean((predict(A,f.x)[mask]-Y[mask])**2)),f.x)
 for j in 'xy':
  n=s['n'+j]; h=s['L'+j]/(n-key[-1]); k=s['kappa_'+j][[0,-1]]; v=s['vel_'+j][[0,-1]]; r=s['robin_'+j]; print(j,'h,k,v,r',h,k,v,r,'r/h',r/h,'r/h^2',r/h**2,'kr/h^2',k*r/h**2,'series',r/h/(1+r*h/(2*k)))
 print('sensor locations',s['sensor_ij'].tolist())
