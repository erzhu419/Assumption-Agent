import numpy as np, pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; X,Y=c['ref_rosette'],c['raw_rosette']; axis=np.count_nonzero(X,axis=1)==1
# Gross rows are separated from every physical reading by nearly an order of magnitude.
clean=np.max(abs(Y-np.median(Y,axis=0)),axis=1)<5
families={'affine':lambda x,k:x,'atan':lambda x,k:np.arctan(k*x)/k,'tanh':lambda x,k:np.tanh(k*x)/k,'asinh':lambda x,k:np.arcsinh(k*x)/k,'sqrt':lambda x,k:x/np.sqrt(1+(k*x)**2),'rational':lambda x,k:x/(1+(k*x)**2),'softsign':lambda x,k:x/(1+abs(k*x)), 'erf':lambda x,k:__import__('scipy').special.erf(k*x)/k}
for name,f in families.items():
 for layout in ('pre','post'):
  if layout=='post' and name=='affine': continue
  def fit(i):
   x,y=X[i],Y[i]
   if layout=='pre':
    def d(xx,p): return np.c_[np.ones(len(xx)),f(xx,np.exp(p))]
    def b(p): return np.linalg.lstsq(d(x,p),y,rcond=None)[0]
    p=least_squares(lambda p:(d(x,p)@b(p)-y).ravel(),np.zeros(4),bounds=(-6,4),gtol=1e-12).x
    B=b(p); return lambda xx:d(xx,p)@B, np.exp(p), B
   def model(xx,p): return p[:4]+f(xx@p[4:20].reshape(4,4),np.exp(p[20:]))
   p=least_squares(lambda p:(model(x,p)-y).ravel(),np.r_[np.zeros(4),np.eye(4).ravel()*1.8,np.zeros(4)],bounds=(np.r_[np.full(20,-np.inf),np.full(4,-6)],np.r_[np.full(20,np.inf),np.full(4,4)]),max_nfev=150).x
   return lambda xx:model(xx,p),np.exp(p[20:]),p[4:20].reshape(4,4)
  pr,k,B=fit(clean); err=pr(X)-Y; cv=np.zeros_like(Y); idx=np.flatnonzero(clean); rng=np.random.default_rng(41); rng.shuffle(idx)
  for test in np.array_split(idx,5):
   tr=clean.copy();tr[test]=False;cv[test]=fit(tr)[0](X[test])-Y[test]
  print(name,layout,'train',np.round(np.sqrt(np.mean(err[clean]**2,0)),6),'cv axis/mixed',[round(np.sqrt(np.mean(cv[clean&m]**2)),6) for m in (axis,~axis)],'k',np.round(k,4));
  if name in ('atan','sqrt','rational'):print(np.round(B,5))
print('gross rows:',np.where(~clean)[0]); print('clean extremes:',Y[clean].min(0),Y[clean].max(0))
