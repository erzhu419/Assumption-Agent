import numpy as np,pickle,json
from scipy.optimize import least_squares,minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; C=s['sensor_calibration_sweeps']; X=C['ref_rosette']; Y=C['raw_rosette']; clean=np.max(abs(Y),axis=1)<4; X,Y=X[clean],Y[clean]; axis=np.count_nonzero(X,axis=1)==1; q=C['single_bank_sweeps']; bank=q['bank_id']; sig=q['ref_signal']; raw=q['raw_readout'];
for j in range(4):
 o=(j+2)%4; a,b=X[:,j],X[:,o]; offset=np.median(Y[axis&(a==0)&(b==0),j]); m=axis&(a!=0); n=(bank==j)&(abs(raw)<4); t=np.r_[a[m],sig[n]]; z=np.r_[Y[m,j],raw[n]]-offset
 p=least_squares(lambda p:p[0]*t+p[1]*np.sin(p[2]*t)-z,[1,.25,2.5]).x; own=lambda v:p[0]*v+p[1]*np.sin(p[2]*v); r=Y[:,j]-offset-own(a); print('CHANNEL',j,'own',p.round(6),'offset',offset,'own max',max(abs(own(t)-z))); print('opposite',np.c_[b[axis&(b!=0)],r[axis&(b!=0)]].round(6).tolist()); print('other axes max',max(abs(r[axis&(a==0)&(b==0)])))
 c,d=X[:,(j+1)%4],X[:,(j-1)%4];
 for name,D in [('poly2',np.c_[b,b*b,a*b,c*d,a*c,a*d,b*c,b*d,c,c*c,d,d*d]),('poly3',np.c_[b,b*b,a*b,c*d,a*c,a*d,b*c,b*d,c,c*c,d,d*d,b**3,a*a*b,a*b*b,a*c*d,b*c*d]),('sin-poly',np.c_[b,b*b,a*b,c*d,a*c,a*d,b*c,b*d,c,c*c,d,d*d,np.sin(p[2]*b),np.sin(p[2]*(a+b))-np.sin(p[2]*a)-np.sin(p[2]*b),np.cos(p[2]*a)*b-b,np.sin(p[2]*a)*b,np.sin(p[2]*b)*a])]:
  beta=np.linalg.lstsq(D,r,rcond=None)[0]; err=[]; fold=np.arange(len(r))%4
  for f in range(4):
   fit=fold!=f; err.extend((D[~fit]@np.linalg.lstsq(D[fit],r[fit],rcond=None)[0]-r[~fit]).tolist())
  print(name,np.round(beta,6), 'cv',np.sqrt(np.mean(np.square(err))),max(abs(D@beta-r)))
  if name=='poly2': print('top residuals',np.c_[a,b,c,d,r,r-D@beta][np.argsort(abs(r-D@beta))[-5:]].round(5).tolist())
for e in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 x=np.array(e['facility_telemetry_trace']); print('TRACE',e['completed_campaign_short_name'],x.shape, 'start',np.round(x[:4],3).tolist(),'mid',np.round(x[len(x)//2],3).tolist(),'end',np.round(x[-4:],3).tolist(),'stdtail',np.round(x[-10:].std(0),3).tolist())
