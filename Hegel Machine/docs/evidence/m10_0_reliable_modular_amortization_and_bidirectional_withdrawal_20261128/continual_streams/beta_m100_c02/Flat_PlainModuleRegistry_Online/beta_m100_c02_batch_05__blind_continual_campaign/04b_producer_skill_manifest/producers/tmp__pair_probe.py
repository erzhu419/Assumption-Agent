import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=map(np.asarray,(c['ref_rosette'],c['raw_rosette'])); clean=np.max(abs(y),axis=1)<5
b=c['single_bank_sweeps']; u,v,j=map(np.asarray,(b['ref_signal'],b['raw_readout'],b['bank_id'])); ok=abs(v)<5; u,v,j=u[ok],v[ok],j[ok]; n=len(u); B=np.eye(4)[j]; folds=np.random.default_rng(8).permutation(len(x))%5
print('rosette dimensions and ranges', x.shape, np.max(abs(x),axis=0)); print('pulse key', [k for k in s if 'pulse' in k]);
def fit_shape(f,starts,lo,hi):
 def res(p):
  D=np.c_[B,B*f(u,p)[:,None]]; return D@np.linalg.lstsq(D,v,rcond=None)[0]-v
 trials=[least_squares(res,p,bounds=(lo,hi),max_nfev=600,gtol=1e-13) for p in starts]; return min(trials,key=lambda q:np.linalg.norm(q.fun))
models={
 'tanh_sin':(lambda u,p:np.tanh(p[0]*u)+p[1]*np.sin(p[2]*u),[[a,b,c] for a,b,c in [(0.68,-.73,.74),(.9,.15,2),(.9,-.15,2),(1,.1,4),(1,-.1,4),(1,.05,6),(1,-.05,6)]],[.05,-10,.05],[8,10,12]),
 'tanh_rat':(lambda u,p:np.tanh(p[0]*u)+p[1]*u/(1+p[2]*u*u),[[a,b,c] for a in [.4,1.2,2.] for b in [-.7,.25] for c in [.1,1.]],[.05,-10,0],[8,10,20]),
 'tanh_over_rat':(lambda u,p:np.tanh(p[0]*u)*(1+p[1]*u*u)/(1+p[2]*u*u),[[a,b,c] for a in [.5,1.2,2.] for b,c in [(0.02,.3),(.3,.02),(1.,1.)]],[.05,0,0],[8,20,20]),
 'tanh_sin_rat':(lambda u,p:(np.tanh(p[0]*u)+p[1]*np.sin(p[2]*u))/(1+p[3]*u*u),[[.68,-.73,.74,.01],[1,.15,2,.1],[1,-.15,2,.1],[1,.08,4,.1]], [.05,-10,.05,-.2],[8,10,12,8]),
 'tanh_cubic':(lambda u,p:np.tanh(p[0]*u)+p[1]*u+p[2]*u**3,[[.72,-.53,.048]], [.05,-10,-2],[8,10,2]),
}
best=None
for name,(f,starts,lo,hi) in models.items():
 q=fit_shape(f,starts,lo,hi); print(name, np.sqrt(np.mean(q.fun**2)),q.x)
 if best is None or np.linalg.norm(q.fun)<best[0]:best=(np.linalg.norm(q.fun), f,q.x)
_,f,p=best
z=f(x,p); Dz=np.c_[B,B*f(u,p)[:,None]]; coeff=np.linalg.lstsq(Dz,v,rcond=None)[0]; diag=coeff[:4]+z*coeff[4:]; rem=y-diag
print('bank bias/gain',coeff.reshape(2,4)); print('raw minus independently predicted self response, first 26 clean rows'); print(np.round(np.c_[x,rem][clean][:26],5))
for mode in ('pair_quadratic','pair_cubic','pair_transformed','pair_rational','pair_sine','all_channels'):
 errors=[]
 for fold in range(5):
  train=clean&(folds!=fold); test=clean&(folds==fold); pred=[]
  for j0 in range(4):
   a=x[:,j0]; w=x[:,(j0+2)%4]; za=z[:,j0]; zw=z[:,(j0+2)%4]; D=np.c_[np.ones(len(x)),za,w,w*w,a*w]
   if mode=='pair_cubic':D=np.c_[D,w**3,a*w*w,a*a*w]
   if mode=='pair_transformed':D=np.c_[np.ones(len(x)),za,zw,zw*zw,za*zw,w, w*w]
   if mode=='all_channels': D=np.c_[np.ones(len(x)),za,x,x*x]
   if mode in ('pair_rational','pair_sine'):
    def factory(t):
     q=w/(1+t[0]*w*w) if mode=='pair_rational' else np.sin(t[0]*w)/t[0]; return np.c_[np.ones(len(x)),za,q,q*q,a*q]
    def err(t):
     E=factory(t)[train]; return E@np.linalg.lstsq(E,y[train,j0],rcond=None)[0]-y[train,j0]
    D=factory(least_squares(err,[.4],bounds=(.0001,2.)).x)
   pred.append(D[test]@np.linalg.lstsq(D[train],y[train,j0],rcond=None)[0]);
  errors.append(np.sqrt(np.mean((np.array(pred).T-y[test])**2)))
 print(mode, np.mean(errors),max(errors))
