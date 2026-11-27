import pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; sw=s['sensor_calibration_sweeps']; x,y=sw['ref_rosette'],sw['raw_rosette']; j=np.arange(4); n=len(x)
def design(x):
 f=x[:,(j+1)%4]; p=x[:,(j-1)%4]; return np.stack((np.ones_like(x),x,f,p,x**3,x*f),axis=-1)
def fit(x,y,shared):
 z=design(x); b=np.stack([np.linalg.lstsq(z[:,i],y[:,i],rcond=None)[0] for i in j]);
 if shared:
  q=np.mean(b[:,2:]/b[:,1,None],axis=0)
  for _ in range(20):
   h=z[:,:,1]+z[:,:,2:]@q; g=np.sum((h-h.mean(0))*(y-y.mean(0)),axis=0)/np.sum((h-h.mean(0))**2,axis=0); a=(y-h*g).mean(0); q=np.linalg.lstsq((z[:,:,2:]*g[None,:,None]).reshape(-1,4),(y-a-g*z[:,:,1]).ravel(),rcond=None)[0]
  b=np.column_stack((a,g,g[:,None]*q))
 return b
def forward(x,b): return np.einsum('nij,ij->ni',design(x),b)
def inverse(y,b):
 x=np.zeros_like(y)
 for k in range(12):
  J=np.zeros((len(x),4,4)); J[:,j,j]=b[:,1]+3*b[:,4]*x*x+b[:,5]*x[:,(j+1)%4]; J[:,j,(j+1)%4]=b[:,2]+b[:,5]*x; J[:,j,(j-1)%4]=b[:,3]; x-=np.linalg.solve(J,(forward(x,b)-y)[...,None])[...,0]
 return x
clean=np.max(abs(y),axis=1)<5
for shared in (False,True):
 b=fit(x[clean],y[clean],shared); print('shared',shared,'b',b,'ratios',b[:,2:]/b[:,1,None]); err=np.zeros_like(x); fer=np.zeros_like(y)
 for f in range(5):
  te=clean&(np.arange(n)%5==f); tr=clean&~te; b=fit(x[tr],y[tr],shared); err[te]=inverse(y[te],b)-x[te]; fer[te]=forward(x[te],b)-y[te]
 for label,t in [('axis',clean & (np.count_nonzero(x,axis=1)==1)),('mixed',clean & (np.count_nonzero(x,axis=1)>1)),('central',clean & (np.max(abs(x),axis=1)<.7)),('tails',clean & (np.max(abs(x),axis=1)>1.1))]:
  if t.any(): print(label,t.sum(),'inverse max and RMS',np.max(abs(err[t])),np.sqrt(np.mean(err[t]**2,axis=0)),'forward max',np.max(abs(fer[t])))
 print('obs inversion range',np.min(inverse(s['y_obs_raw'].reshape(-1,4),b),axis=0),np.max(inverse(s['y_obs_raw'].reshape(-1,4),b),axis=0))
