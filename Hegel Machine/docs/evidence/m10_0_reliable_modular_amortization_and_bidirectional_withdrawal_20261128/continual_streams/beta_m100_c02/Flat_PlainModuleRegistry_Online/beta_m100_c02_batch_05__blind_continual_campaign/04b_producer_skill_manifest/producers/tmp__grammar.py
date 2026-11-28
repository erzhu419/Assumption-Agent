import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; clean=np.max(abs(y),axis=1)<5
b=c['single_bank_sweeps']; bx=b['ref_signal']; by=b['raw_readout']; bank=b['bank_id']; good=abs(by)<5
B=np.eye(4)[bank]; print('single bank shared shape offsets at zero')
forms={
 'tanh':lambda x,a:np.tanh(a*x)/a,
 'atan':lambda x,a:np.arctan(a*x)/a,
 'sqrt':lambda x,a:x/np.sqrt(1+a*x*x),
 'rat':lambda x,a:x/(1+a*x*x),
}
extras={'none':lambda x,a,z:[], 'linear':lambda x,a,z:[x], 'cubic':lambda x,a,z:[x**3], 'square':lambda x,a,z:[x*x], 'shapesq':lambda x,a,z:[z*z], 'cubic_rat':lambda x,a,z:[x**3/(1+x*x)], 'tanh_cub':lambda x,a,z:[z**3], 'tanh_cub_square':lambda x,a,z:[z**3,z*z], 'linear_square':lambda x,a,z:[x,z*z], 'poly_rational':lambda x,a,z:[x,x/(1+x*x),z*z]}
for key,f in forms.items():
 for name,extra in extras.items():
  def design(a):
   z=f(bx,a[0]);return np.hstack([B]+[B*w[:,None] for w in [z]+extra(bx,a,z)])
  def resid(a):
   d=design(a)[good]; return d@np.linalg.lstsq(d,by[good],rcond=1e-11)[0]-by[good]
  sol=least_squares(resid,[1.],bounds=(.02,5),gtol=1e-13);print(key,name,round(sol.x[0],6),format(np.std(sol.fun),'.4g'))
print('axis data');print(np.round(np.c_[x,y][:16],5));print('mixed data model comparison')
fold=np.random.default_rng(8).permutation(len(x))%5
pairs=list(combinations(range(4),2));triples=list(combinations(range(4),3))
for name,f in forms.items():
 for mode in range(7):
  def design(x,a):
   z=f(x,a[0]); cols=[np.ones(len(x)), z,z*z]
   if mode in [1,3]:cols += [z[:,i]*z[:,j] for i,j in pairs]
   if mode in [2,4]:cols += [x[:,i]*x[:,j] for i,j in pairs]
   if mode in [3,4]:cols += [z[:,i]*z[:,j]*z[:,k] for i,j,k in triples]
   if mode==5:cols += [z**3]+[z[:,i]*z[:,j] for i,j in pairs]
   if mode==6:cols += [x]+[z[:,i]*z[:,j] for i,j in pairs]
   return np.column_stack(cols)
  errors=[]
  for k in range(5):
   tr=clean&(fold!=k); te=clean&(fold==k)
   def coef(a):return np.linalg.lstsq(design(x[tr],a),y[tr],rcond=1e-11)[0]
   sol=least_squares(lambda a:(design(x[tr],a)@coef(a)-y[tr]).ravel(),[1.],bounds=(.02,5)); errors.append(np.sqrt(np.mean((design(x[te],sol.x)@coef(sol.x)-y[te])**2)))
  print(name,mode,round(np.mean(errors),6),round(max(errors),6))
