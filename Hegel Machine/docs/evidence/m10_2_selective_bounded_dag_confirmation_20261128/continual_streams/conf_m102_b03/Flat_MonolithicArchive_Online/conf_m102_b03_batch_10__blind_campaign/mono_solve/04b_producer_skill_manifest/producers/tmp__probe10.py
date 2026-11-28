import numpy as np,pickle
from scipy.optimize import minimize_scalar,least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; ls=lambda d,y:np.linalg.lstsq(d,y,rcond=1e-10)[0]; D=lambda x:np.c_[np.ones(len(x)),x];
a,b=s['panel_a']['u'],s['panel_a']['v']; i,j=np.triu_indices(len(a),1); slope=np.median((b[i]-b[j])/(a[i]-a[j]),axis=0); r=b-a*slope; good=np.linalg.norm(r-np.median(r,0),axis=1)<2; A=ls(D(a[good]),b[good]); print('calibration',good.sum(),A,np.std(b[good]-D(a[good])@A,0)); B=ls(D(s['panel_b']['u']),s['panel_b']['v']); print('B',B)
h=lambda p:-np.expm1(-t[:,None]*p); Y=tr['traj']; p=least_squares(lambda p:(h(p)@ls(h(p),Y.T)-Y.T).ravel(),[2.,.45],bounds=([.65,.02],[8,1.5])).x; print('time',p,np.std(h(p)@ls(h(p),Y.T)-Y.T)); target=np.c_[tr['y'],ls(h(p),Y.T).T]; print('amps std',target.std(0),'ranges',target.min(0),target.max(0)); u=s['panel_c']['u']; z=(tr['x']-A[0])@np.linalg.inv(A[1:]);
F={'linear':lambda x,k:x,'square':lambda x,k:x*x,'edge':lambda x,k:x*np.roll(x,-1,axis=1),'cos':lambda x,k:np.cos(k*x),'sin':lambda x,k:np.sin(k*x),'tanh':lambda x,k:np.tanh(k*x),'softplus':lambda x,k:np.logaddexp(0,k*x),'softsign':lambda x,k:x/(1+k*abs(x)),'gauss':lambda x,k:np.exp(-k*x*x),'abs':lambda x,k:abs(x),'rational':lambda x,k:x/(1+k*x*x),'evenrat':lambda x,k:x*x/(1+k*x*x),'odd3':lambda x,k:x**3,'sincos':lambda x,k:np.c_[np.sin(k*x),np.cos(k*x)],'interaction':lambda x,k:(x[:,:,None]*x[:,None,:])[:,np.triu_indices(4,1)[0],np.triu_indices(4,1)[1]]}
for name in ['v1','v2']:
 v=s['panel_c'][name]; out=[]
 for f,func in F.items():
  for cat in (False,True):
   def mat(x,k):return D(np.c_[x,func(x,k)] if cat else func(x,k))
   def loss(k):
    M=mat(u,k); return np.mean((M@ls(M,v)-v)**2)
   rr=minimize_scalar(loss,bounds=(.08,5),method='bounded'); out.append((rr.fun**.5,f,cat,rr.x))
 print(name,sorted(out)[:12]);
