import pickle,json,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['sensor_calibration_sweeps']; X,Y=a['ref_rosette'],a['raw_rosette']; b=a['single_bank_sweeps']; j=(b['bank_id']==1); t=b['ref_signal'][j]; y=b['raw_readout'][j]; np.set_printoptions(precision=7,suppress=True)
models={
 'rational': (lambda x,p: np.array([x/(1+p[0]*x*x),x**3/(1+p[0]*x*x)]).T,[.6]),
 'rational4': (lambda x,p: np.array([x/(1+p[0]*x*x+p[1]*x**4),x**3/(1+p[0]*x*x+p[1]*x**4)]).T,[.6,.05]),
 'rational5': (lambda x,p: np.array([x/(1+p[0]*x*x+p[1]*x**4),x**3/(1+p[0]*x*x+p[1]*x**4),x**5/(1+p[0]*x*x+p[1]*x**4)]).T,[.6,.05]),
 'power': (lambda x,p: np.array([x/(1+p[0]*x*x)**p[1],x]).T,[.6,1.]),
 'powerodd': (lambda x,p: np.array([x/(1+p[0]*x*x)**p[1],x**3/(1+p[0]*x*x)**p[1]]).T,[.6,1.]),
 'exp': (lambda x,p: np.array([x*np.exp(-p[0]*x*x),x]).T,[.5]),
 'tanh+rat': (lambda x,p: np.array([np.tanh(p[0]*x),x/(1+p[1]*x*x)]).T,[1.,.5]),
 'tanh+tan': (lambda x,p: np.array([np.tanh(p[0]*x),np.tanh(p[1]*x)]).T,[1.3,.7]),
 'cheby9': (lambda x,p: np.array([x**k for k in (1,3,5,7,9)]).T,[])}
def fit(x,y,f,p):
 def calc(q,z): return np.c_[np.ones(len(z)),f(z,q)]
 def resid(q):
  A=calc(q,x); return A@np.linalg.lstsq(A,y,rcond=None)[0]-y
 q=least_squares(resid,p,bounds=(.00001,10),max_nfev=600,gtol=1e-13).x if len(p) else p
 A=calc(q,x); c=np.linalg.lstsq(A,y,rcond=None)[0]; return lambda z:calc(q,z)@c,q,c
fits=[]
for name,(f,p) in models.items():
 fun,q,c=fit(t,y,f,p); loo=max(abs(float(fit(t[np.arange(len(t))!=i],y[np.arange(len(t))!=i],f,p)[0](t[i:i+1])[0])-y[i]) for i in range(len(t))); print(name,max(abs(fun(t)-y)),loo,q,c); fits.append((loo,name,fun))
_,name,fun=min(fits,key=lambda r:r[0]); print('best',name,'single bank residual',fun(t)-y)
for i in range(4):
 j=(b['bank_id']==i)&(abs(b['raw_readout'])<5); t=b['ref_signal'][j]; v=b['raw_readout'][j]; off,gain=np.linalg.lstsq(np.c_[np.ones(len(t)),fun(t)-fun(np.array([0]))],v,rcond=None)[0]; keep=abs(Y[:,i])<5; x=X[keep]; scale=gain; e=(Y[keep,i]-off)/scale-(fun(x[:,i])-fun(np.array([0]))); o=x[:,(i+2)%4]; ii=np.argsort(o); print('opposite curve',i,'gain',gain,np.c_[o[ii],e[ii]][::5].round(5));
log=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for r in log:
 T=np.array(r['facility_telemetry_trace']); print(r['completed_campaign_short_name'],T.mean(0).round(3),T.std(0).round(3))
