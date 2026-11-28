import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=[s['panel_a'][k] for k in ('u','v')]; ix=np.arange(4); jx=(ix+2)%4
print('telemetry',s['telemetry_vec'],'raw range',s['train']['x'].min(0),s['train']['x'].max(0))
def design(u,q,mode,j):
 a,b=u[:,j],u[:,jx[j]]; cols=[np.ones(len(u)),a];
 if mode=='fixedplus': return np.column_stack(cols+[np.sin(2.15*a),np.sin(2.15*b),np.sin(q[0]*b)])
 if mode=='sinlin': cols.append(b)
 if mode=='rotate': return np.column_stack(cols+[np.sin(q[0]*a),np.sin(q[1]*(b+q[2]*a))])
 if mode=='tanh': return np.column_stack(cols+[np.tanh(q[0]*a),np.tanh(q[1]*b)])
 return np.column_stack(cols+[np.sin(q[0]*a),np.sin(q[1]*b)])
def fits(q,mode,mask=None):
 cs=[]; err=[]
 for j in ix:
  X=design(u,q,mode,j); y=v[:,j]; ok=np.ones(len(u),bool) if mask is None else mask[:,j]; w=ok.astype(float)
  for _ in range(22 if mask is None else 1):
   c=np.linalg.lstsq(X*w[:,None],y*w,rcond=None)[0]; e=X@c-y
   if mask is None: w=(1+(e/.008)**2)**(-.25)
  cs.append(c); err.append(e)
 return np.array(cs),np.stack(err,1)
for mode in ['sin','sinlin','tanh','rotate','fixedplus']:
 q0={'tanh':[1,1],'rotate':[2.15,2.45,0],'fixedplus':[2.45]}.get(mode,[2.15,2.45]); z=least_squares(lambda q:fits(q,mode)[1].ravel(),q0,loss='soft_l1',f_scale=.01,max_nfev=85); mask=abs(z.fun.reshape(v.shape))<.1
 z=least_squares(lambda q:fits(q,mode,mask)[1][mask],z.x,max_nfev=100); c,e=fits(z.x,mode,mask); print(mode,z.x,'RMS',np.sqrt(np.mean(e[mask]**2)), 'max inlier',abs(e[mask]).max(),'clean',mask.sum(),'coefs',c.round(6))
