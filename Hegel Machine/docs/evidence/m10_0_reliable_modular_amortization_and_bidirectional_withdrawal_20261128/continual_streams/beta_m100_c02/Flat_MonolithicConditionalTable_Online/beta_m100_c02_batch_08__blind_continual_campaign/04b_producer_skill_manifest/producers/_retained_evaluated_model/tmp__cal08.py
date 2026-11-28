import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

def data(s):
 d=s['sensor_calibration_sweeps']; X,Y=np.asarray(d['ref_rosette']),np.asarray(d['raw_rosette']); good=(abs(Y)<5).all(1); X,Y=X[good],Y[good]; D=d['single_bank_sweeps']; ids,t,z=map(np.asarray,(D['bank_id'],D['ref_signal'],D['raw_readout'])); return X,Y,ids[abs(z)<5],t[abs(z)<5],z[abs(z)<5]

families={
 'rat11': (lambda x,q:[x/(1+q[0]*x*x),x], [1.]),
 'rat21': (lambda x,q:[x/(1+q[0]*x*x),x/(1+q[1]*x*x),x], [.5,2.]),
 'rat22': (lambda x,q:[x/(1+q[0]*x*x+q[1]*x**4),x**3/(1+q[0]*x*x+q[1]*x**4),x], [1.,.2]),
 'erf5': (lambda x,q:[erf(q[0]*x),x,x**3,x**5], [.7]),
 'tanh2': (lambda x,q:[np.tanh(q[0]*x),np.tanh(q[1]*x),x], [1.,2.]),
 'gauss': (lambda x,q:[x*np.exp(-q[0]*x*x),x], [1.]),
 'asinh': (lambda x,q:[np.arcsinh(q[0]*x),x,x/(1+q[1]*x*x)], [1.,1.]),
 'ratpow': (lambda x,q:[x/(1+q[0]*x*x)**q[1],x], [1.,1.]),
 'atan2': (lambda x,q:[np.arctan(q[0]*x),np.arctan(q[1]*x),x], [1.,2.]),
 'soft2': (lambda x,q:[x/np.sqrt(1+q[0]*x*x),x/np.sqrt(1+q[1]*x*x),x], [1.,2.]),
 'rat23': (lambda x,q:[x/(1+q[0]*x*x+q[1]*x**4),x**3/(1+q[0]*x*x+q[1]*x**4),x,x**3], [1.,.2]),
 'tanh_atan': (lambda x,q:[np.tanh(q[0]*x),np.arctan(q[1]*x),x], [1.,1.]),
}

def fit_own(x,y,name):
 f,p0=families[name]; scale=np.max(abs(x));
 def var(q):
  A=np.column_stack([np.ones_like(x),*f(x,q)]); c=np.linalg.lstsq(A,y,rcond=None)[0]; return A@c-y,c
 q=least_squares(lambda q:var(q)[0],p0,bounds=(.00001,25),max_nfev=500,gtol=1e-13).x
 return q,var(q)[1]

if __name__=='__main__':
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; X,Y,ids,t,z=data(s); axis=np.count_nonzero(X,axis=1)==1
 for i in [0,1]:
  ix=axis&(X[:,i]!=0); a=np.r_[X[ix,i],t[ids==i]]; b=np.r_[Y[ix,i],z[ids==i]]; print('BANK',i)
  for name,(f,_) in families.items():
   q,c=fit_own(a,b,name); pred=lambda q,c,x:np.column_stack([np.ones_like(x),*f(x,q)])@c
   train=np.sqrt(np.mean((pred(q,c,a)-b)**2)); err=[]
   for j in range(len(b)):
    k=np.arange(len(b))!=j; qq,cc=fit_own(a[k],b[k],name); err.append(pred(qq,cc,a[~k])[0]-b[j])
   print(name,'train/loo',round(train,7),round(np.std(err),7),'q',q.round(5),'coef',c.round(6))
 for i in range(4):
  j=(i+2)%4; k=axis&(X[:,j]!=0); print('opp',i,np.c_[X[k,j],Y[k,i]].round(6).tolist())
 print('clean mixed head',np.c_[X[~axis],Y[~axis]][:8].round(5)); print('study scalars',[(k,v) for k,v in s.items() if np.isscalar(v)]); print('public items',len(pickle.load(open('/holdout_task/public_batch.pkl','rb'))))
