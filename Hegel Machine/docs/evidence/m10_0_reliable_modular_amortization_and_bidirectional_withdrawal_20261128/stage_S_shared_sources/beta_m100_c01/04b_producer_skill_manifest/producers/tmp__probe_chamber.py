import pickle
import numpy as np
from scipy.linalg import null_space
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
c=s['sensor_calibration_sweeps']; ref=c['ref_rosette']; raw=c['raw_rosette']; design=np.c_[np.ones(len(ref)),ref]
e=[]
for i in range(len(ref)):
 keep=np.arange(len(ref))!=i
 e.append(raw[i]-design[i]@np.linalg.lstsq(design[keep],raw[keep],rcond=None)[0])
a=np.linalg.lstsq(design,raw,rcond=None)[0]
print('calibration intercept and matrix:\n',a,'LOO worst channel rms',np.max(np.mean(np.square(e),axis=0)**.5),'identity rms',np.mean((raw-ref)**2)**.5)
y=np.linalg.solve(a[1:].T,(s['y_obs_raw'].reshape(-1,4)-a[0]).T).T.ravel(); z=s['Z_nuisance']
nx,ny=s['nx'],s['ny']; ij=s['sensor_ij']; slots=ij[:,0]+nx*ij[:,1]; modes=s['source_modes']; n=len(modes)
rhs=modes.transpose(0,2,1).reshape(n,-1).T; w0=np.full(n,s['total_budget']/n); Q=null_space(np.ones((1,n))); idx=np.arange(len(y))
print('y rms',np.mean(y**2)**.5,'source ranges',modes.min(),modes.max(),'Z rms',np.mean(z**2,axis=0)**.5)
def axis(k,v,h,r,rob,wall,opp):
 kp=(k[1:-1]+k[2:])*.5; km=(k[1:-1]+k[:-2])*.5
 lower=np.r_[-km/h**2-np.maximum(v[1:-1],0)/h,-wall[1]*k[-1]/h**2]; upper=np.r_[-wall[0]*k[0]/h**2,-kp/h**2-np.maximum(-v[1:-1],0)/h]
 mid=np.r_[rob[0]*r/h-upper[0]+opp[0]*k[0]/h**2,(km+kp)/h**2+abs(v[1:-1])/h,rob[1]*r/h-lower[-1]+opp[1]*k[-1]/h**2]
 A=diags((lower,mid,upper),(-1,0,1),format='lil'); A[0,-1]=-opp[0]*k[0]/h**2; A[-1,0]=-opp[1]*k[-1]/h**2
 return A.tocsc()
def forward(p):
 p=np.asarray(p).reshape(3,2,2)
 A=[]
 for j,(ax,k) in enumerate(zip('xy',(nx,ny))):
  h=s['L'+ax]/(k-1); A.append(axis(s['kappa_'+ax],s['vel_'+ax],h,s['robin_'+ax],*p[:,j,:]))
 L=kron(eye(ny),A[0])+kron(A[1],eye(nx))+diags(s['V_reaction_2d'].ravel(order='F')); E=eye(nx*ny,format='csc')
 return np.vstack([splu((L+mu*E).tocsc()).solve(rhs)[slots] for mu in s['mu_shifts']])
def fit(S,keep):
 D=np.c_[S@Q,z]; b=y-S@w0; x=np.linalg.lstsq(D[keep],b[keep],rcond=None)[0]
 return D@x-b,w0+Q@x[:n-1]
def report(tag,p):
 S=forward(p); e,w=fit(S,idx); slices=[idx//4==j for j in range(len(y)//4)]; loo=[np.mean(fit(S,~j)[0][j]**2)**.5 for j in slices]; shifts=[np.mean(t**2)**.5 for t in np.split(e,len(s['mu_shifts']))]
 print(tag,'res',np.mean(e**2)**.5,'worst group',max(loo),'shifts',shifts,'w',np.round(w,4))
for tag,p in [('open',[1]*8+[0]*4),('reflect',[0]*4+[1]*4+[0]*4),('double-inward',[1]*4+[2]*4+[0]*4),('half-robin',[.5]*4+[1]*4+[0]*4),('double-robin',[2]*4+[1]*4+[0]*4),('periodic',[0]*4+[1]*8),('leaky-periodic',[1]*12)]:
 report(tag,p)
base=np.array([1.]*8+[0.]*4)
for label,sel in [('robin',np.arange(4)),('robin+inward',np.arange(8)),('all',np.arange(12))]:
 def params(x):
  p=base.copy();p[sel]=x;return p
 res=least_squares(lambda x:fit(forward(params(x)),idx)[0],base[sel],bounds=(0,20),max_nfev=180,gtol=1e-12)
 report(label,params(res.x)); print('wall parameters [robin, inward, wrap]',np.round(params(res.x),5))
