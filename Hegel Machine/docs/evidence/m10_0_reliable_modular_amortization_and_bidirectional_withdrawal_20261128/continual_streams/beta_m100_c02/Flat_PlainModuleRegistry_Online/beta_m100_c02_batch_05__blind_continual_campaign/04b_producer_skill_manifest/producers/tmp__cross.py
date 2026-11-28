import pickle,sys,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from upstream_transducer import _fit,_response
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']['sensor_calibration_sweeps']; x,y=(s[k] for k in ('ref_rosette','raw_rosette')); ok=np.max(abs(y),axis=1)<5
sh,c=_fit(s); z=_response(x,sh); q=np.roll(x,2,axis=1); w=np.roll(z,2,axis=1); folds=np.random.default_rng(8).permutation(len(x))%5
forms={
'baseline':(lambda q,w,p:[w,w*w,q,q*q],[]),
'poly4':(lambda q,w,p:[q,q*q,q**3,q**4],[]),
'sin-even':(lambda q,w,p:[q,np.sin(p[0]*q),1-np.cos(p[0]*q)],[[f] for f in (1.2,2.5,4,6)]),
'sin-quad':(lambda q,w,p:[q,q*q,np.sin(p[0]*q)],[[f] for f in (1.2,2.5,4,6)]),
'cos-quad':(lambda q,w,p:[q,q*q,1-np.cos(p[0]*q)],[[f] for f in (1.2,2.5,4,6)]),
'poly-sin':(lambda q,w,p:[q,q*q,np.sin(p[0]*q),1-np.cos(p[0]*q)],[[f] for f in (1.2,2.5,4,6)]),
'double-sin':(lambda q,w,p:[q,q*q,np.sin(p[0]*q),1-np.cos(p[1]*q)],[[f,g] for f in (1.2,2.5,4) for g in (1.2,2.5,4)]),
'rat-shared':(lambda q,w,p:[q,q*q,q/(1+p[0]*q*q),q*q/(1+p[0]*q*q)],[[.5],[2],[4]]),
'rat-separate':(lambda q,w,p:[q,q*q,q/(1+p[0]*q*q),q*q/(1+p[1]*q*q)],[[.5,.5],[.5,3],[3,.5],[3,3]]),
'atan-tanh':(lambda q,w,p:[q,np.arctan(p[0]*q),np.tanh(p[1]*q)**2],[[.5,.5],[1.5,1.5],[3,3]]),
'rat-transformed':(lambda q,w,p:[q,q*q,w/(1+p[0]*w*w),w*w/(1+p[1]*w*w)],[[.5,.5],[.5,3],[3,.5],[3,3]]),
'linear-rational':(lambda q,w,p:[q,q*q/(1+p[0]*q*q)],[[.5],[2]]),
'sin-rational':(lambda q,w,p:[q,np.sin(p[0]*q),q*q/(1+p[1]*q*q)],[[2.5,.5],[4,.5],[4,2]]),
'quad-tanh':(lambda q,w,p:[q,q*q,np.tanh(p[0]*q)],[[.5],[2]]),
}
for name,(fun,starts) in forms.items():
 rms=[]; inv=[]
 for k in range(6):
  tr=ok&(folds!=k); te=ok&(folds==k) if k<5 else ok
  def calc(p):
   D=np.stack([np.ones_like(z),z]+fun(q,w,p)+([z*w] if name=='baseline' else []),axis=-1)
   b=np.array([np.linalg.lstsq(D[tr,i],y[tr,i],rcond=None)[0] for i in range(4)])
   return np.sum(D*b,axis=-1)-y,b
  p=min((least_squares(lambda p:calc(p)[0][tr].ravel(),p,bounds=(.03,9),max_nfev=100) for p in starts),key=lambda a:a.cost).x if starts else []
  e,b=calc(p); rms.append(np.sqrt(np.mean(e[te]**2)))
 print(name,'worst/full',np.round([max(rms[:5]),rms[-1]],8),'shape',np.round(p,5),'row0',np.round(b[0],5))
# Normalize independent sweep offsets/gains; cross trace on mixed rows estimates the pair law.
single=s['single_bank_sweeps']; B=np.eye(4)[single['bank_id']]; u=single['ref_signal']; v=single['raw_readout']; sel=abs(v)<5; D=np.c_[B,B*_response(u,sh)[:,None]]; b,g=np.linalg.lstsq(D[sel],v[sel],rcond=None)[0].reshape(2,4)
r=(y-b)/g-z; idx=np.argsort(q[ok].ravel()); print('cross trace deciles',np.round(np.c_[q[ok].ravel()[idx],r[ok].ravel()[idx]][::20],6))
