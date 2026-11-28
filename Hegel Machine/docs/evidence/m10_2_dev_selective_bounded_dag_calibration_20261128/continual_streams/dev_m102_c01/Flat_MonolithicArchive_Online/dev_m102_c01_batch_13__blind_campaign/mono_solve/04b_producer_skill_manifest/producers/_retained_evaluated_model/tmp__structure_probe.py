import numpy as np,pickle
from panels import calibration,panels,affine,fit
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,T=(s['train'][k] for k in ('x','y','traj')); n=52; p=np.linalg.svd(T[:n],full_matrices=False)[2][0]; A=T@p*p[-1]; g,h=calibration(s['panel_a']); B,c1,c2=panels(s); z=(x-h)/g; b=lambda z:affine(z)@B; bi=lambda z:(z-B[0])@np.linalg.inv(B[1:]); Z={'x':x,'z':z,'b':b(z),'bi':bi(z),'bf':b(x*g+h)}
for k,v in list(Z.items()):
 Z[k+'c1']=c1(v); Z[k+'c2']=c2(v)
def quad(z):
 i,j=np.triu_indices(z.shape[1]);return z[:,i]*z[:,j]
def features(w):
 return {'linear':affine(w),'quad':affine(np.c_[w,quad(w)]),'sat':affine(np.c_[np.tanh(1.25*w),quad(w)]),'trig':affine(np.c_[np.sin(1.25*w),np.cos(1.25*w),quad(w)[:,[1,2,3,5,6,8]]])}
for label,out in [('y',y),('a',A)]:
 results=[]
 for name,w in Z.items():
  for fn,F in features(w).items():
   for model in ('linear','tanh','soft','atan'):
    if model=='linear':
     q=F@fit(F[:n],out[:n]); results.append((np.mean(abs(q-out)[:n]),np.mean(abs(q-out)[n:]),name,fn,model));continue
    for cross in ([False,True] if label=='y' else [False]):
     r=A if cross else np.zeros(len(x)); d=F.shape[1]
     for k in (1.1,1.7):
      def pred(v):
       a=F@v[:d]; scale=np.exp(v[d]); u=a/scale
       if model=='tanh': f=np.tanh(u)
       elif model=='soft': f=u/np.sqrt(1+u*u)
       else:f=np.arctan(u)
       return scale*f+v[d+1]+v[d+2]*r
      v=least_squares(lambda v:(pred(v)-out)[:n],np.r_[fit(F[:n],out[:n]),np.log(k),0.,0.],bounds=(np.r_[np.full(d,-50),-.8,-4,-4],np.r_[np.full(d,50),3,4,4]),max_nfev=110).x
      q=pred(v); results.append((np.mean(abs(q-out)[:n]),np.mean(abs(q-out)[n:]),name,fn,model,cross,np.round(v[-3:],3)))
 for j in (0,1):print(label,j,*sorted(results,key=lambda a:a[j])[:8],sep='\n')
