import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['sensor_calibration_sweeps']; X,Y=a['ref_rosette'],a['raw_rosette']; b=a['single_bank_sweeps']; j=(b['bank_id']==1); t=b['ref_signal'][j]; y=b['raw_readout'][j]; np.set_printoptions(precision=6,suppress=True)
F={'tanh+cubic':(lambda x,p: np.tanh(p[0]*x)+p[1]*x+p[2]*x**3,[1,.1,.01]),'tanh+atan':(lambda x,p: np.tanh(p[0]*x)+p[1]*np.arctan(p[2]*x),[1,.2,.8]),'tanh+erf':(lambda x,p: np.tanh(p[0]*x)+p[1]*erf(p[2]*x),[1,.2,.8]),'tanh+rational':(lambda x,p: np.tanh(p[0]*x)+p[1]*x/(1+p[2]**2*x*x),[1,.2,.8]),'tanh+sqrt':(lambda x,p: np.tanh(p[0]*x)+p[1]*x/np.sqrt(1+p[2]**2*x*x),[1,.2,.8]),'tanh+quintic':(lambda x,p: np.tanh(p[0]*x)+p[1]*x+p[2]*x**3+p[3]*x**5,[1,.1,.01,0]),'tanh(cubic)+linear':(lambda x,p: np.tanh(p[0]*x+p[1]*x**3)+p[2]*x,[1,0,.1]),'tanh+softsign':(lambda x,p: np.tanh(p[0]*x)+p[1]*x/(1+p[2]*abs(x)),[1,.2,.8]),'tanh+sin':(lambda x,p: np.tanh(p[0]*x)+p[1]*np.sin(p[2]*x),[1,.2,.8]),'tanh+tanh':(lambda x,p: np.tanh(p[0]*x)+p[1]*np.tanh(p[2]*x),[1,.2,.5]),'tanh+asinh':(lambda x,p: np.tanh(p[0]*x)+p[1]*np.arcsinh(p[2]*x),[1,.2,.8]),'tanh+linear+relu2':(lambda x,p: np.tanh(p[0]*x)+p[1]*x+p[2]*x*abs(x),[1,.1,0])}
R=[]
for k,(f,p) in F.items():
 fun=lambda q:q[0]+q[1]*f(t,q[2:])-y
 q=least_squares(fun,[.07,1.5]+p,max_nfev=1500).x; R.append((np.linalg.norm(fun(q)),k,q));
for row in sorted(R): print(row)
_,name,p=min(R); f=F[name][0];
for i in range(4):
 j=(b['bank_id']==i)&(abs(b['raw_readout'])<5); t=b['ref_signal'][j]; v=b['raw_readout'][j]; off,gain=np.linalg.lstsq(np.c_[np.ones(len(t)),f(t,p[2:])],v,rcond=None)[0]; keep=abs(Y[:,i])<5; x=X[keep]; e=Y[keep,i]-off-gain*f(x[:,i],p[2:]); o=x[:,(i+2)%4]; l=x[:,(i-1)%4]; r=x[:,(i+1)%4]; z=x[:,i];
 print('bank',i,'scalar',off,gain, 'max',max(abs(v-off-gain*f(t,p[2:]))));
 for expr in [(o,o*o,l,r),(o,o*o,o**3,l,r),(o,o*o,o**3,l,r,z*o,z*l,z*r),(o,o*o,o**3,l,r,z*o,z*l,z*r,l*r,l*l,r*r),(o,np.maximum(o,0),np.maximum(o,0)**2,l,r,z*o,z*l,z*r),(o,np.maximum(o,0),o*abs(o),l,r,z*o,z*l,z*r)]:
  D=np.array(expr).T; c=np.linalg.lstsq(D,e,rcond=None)[0]; print(np.round(c,5),np.quantile(abs(e-D@c),[.5,.95,1]))
