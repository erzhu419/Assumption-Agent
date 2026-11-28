import pickle,numpy as np,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; z=s['y_obs_raw'].reshape(-1,4)
keep=(abs(y)<5).all(1); X,Y=x[keep],y[keep]; print('axis pairs',np.round(np.c_[x,y][:16],6));
np.set_printoptions(precision=5,suppress=True,linewidth=140)
def design(x,k,w,j,ext):
 t=x[:,j]; g=w[0]; f={'tanh':lambda:np.tanh(g*t),'rat':lambda:t/(1+g*t*t),'root':lambda:t/np.sqrt(1+g*t*t),'atan':lambda:np.arctan(g*t),'erf':lambda:__import__('scipy').special.erf(g*t)}[k]()
 a=[np.ones(len(x)),f,*x[:,np.arange(4)!=j].T]
 if 'l' in ext:a.append(t)
 if 'q' in ext:a.extend((x*x).T)
 if 'c' in ext:a.extend((x**3).T[np.arange(4)!=j])
 if 'r' in ext:a.extend([x[:,i]/(1+w[1]*x[:,i]**2) for i in range(4) if i!=j])
 if 'p' in ext:a.extend([x[:,i]*x[:,l] for i,l in itertools.combinations(range(4),2)])
 if 'd' in ext:a.extend([x[:,i]*t*t for i in range(4) if i!=j]+[x[:,i]**2*t for i in range(4) if i!=j])
 return np.array(a).T
def solve(x,y,k,ext):
 out=[]
 for j in range(4):
  def profile(w, ret=False):
   a=design(x,k,w,j,ext); p=np.linalg.lstsq(a,y[:,j],rcond=1e-10)[0]; return p if ret else a@p-y[:,j]
  w=least_squares(profile,[.8,.6] if 'r' in ext else [.8],bounds=(.005,8),ftol=1e-10).x; out.append((w,profile(w,True)))
 return out
def pred(x,k,e,f): return np.column_stack([design(x,k,w,j,e)@b for j,(w,b) in enumerate(f)])
for k in ['tanh','rat','root','atan','erf']:
 for e in ['', 'q','qc','qr','qcp','qcd','lq','lqr']:
  f=solve(X,Y,k,e); r=pred(X,k,e,f)-Y; rms=lambda r:np.sqrt(np.mean(r*r)); sc=[]
  for order in [np.arange(len(X)),np.argsort(np.linalg.norm(X,axis=1)),np.random.default_rng(11).permutation(len(X))]:
   for test in np.array_split(order,5):
    tr=np.setdiff1d(np.arange(len(X)),test); sc.append(rms(pred(X[test],k,e,solve(X[tr],Y[tr],k,e))-Y[test]))
  print(k,e,'fit',round(rms(r),6),'cv',round(max(sc),6),'scales',[np.round(w,4).tolist() for w,b in f]);
  if k=='tanh' and e in ('q','qr','qcd') or rms(r)<.003: print(' coefs',np.array([b for w,b in f])); print(' resid axis',r[:13])
