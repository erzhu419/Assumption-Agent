import pickle, numpy as np, itertools, json, sys
from scipy.optimize import least_squares, root
sys.path[:0]=['/holdout_task','/tmp']; import batch_solver as m
from cal08 import data
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; X,Y,ids,t,z=data(s)
print('study scalars',[(k,v) for k,v in s.items() if np.isscalar(v)])
a=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for e in a:
 q=np.asarray(e['facility_telemetry_trace']);print('trace',e['completed_campaign_short_name'],q.mean(0).round(2))
print('aux first/last',s['aux_manifold_telemetry'][[0,-1]])
def terms(a,b,d): return np.column_stack([a**(i-j)*b**j for i in d for j in range(i+1)])
def fit(X,Y,kind):
 fs=[]
 for i in range(4):
  T=np.vstack((X,np.eye(4)[i]*t[ids==i,None])); y=np.r_[Y[:,i],z[ids==i]]; a,b=T[:,i],T[:,(i+2)%4]
  if kind in ['pade2','pade4']:
   deg=[2] if kind=='pade2' else [2,4]; N=terms(a,b,range(6)); D=terms(a,b,deg); p=np.linalg.lstsq(np.c_[N,-D*y[:,None]],y,rcond=None)[0]; n=N.shape[1]; p=least_squares(lambda p:N@p[:n]/(1+D@p[n:])-y,p,max_nfev=150).x
   def f(a,b,p=p,n=n,deg=deg):return terms(a,b,range(6))@p[:n]/(1+terms(a,b,deg)@p[n:])
  else:
   def design(a,b,q,kind=kind):
    k,r,h,j,l=q; u=a+k*b if kind!='sep' else a; den=1+r*u*u+h*u**4; den2=1+j*b*b+l*b**4
    return np.column_stack([np.ones(len(a)),u/den,u**3/den,u**5/den,b/den2,b*b/den2,b**3/den2,b**4/den2]+([a*b,a*b*b,a*a*b,a*a*b*b] if kind=='inter' else []))
   def res(q):
    A=design(a,b,q); c=np.linalg.lstsq(A,y,rcond=None)[0]; return A@c-y
   q=least_squares(res,[-.12,.23,.03,.25,.02],bounds=([-1,0,0,0,0],[1,4,2,4,2]),max_nfev=100).x; c=np.linalg.lstsq(design(a,b,q),y,rcond=None)[0];
   def f(a,b,q=q,c=c,design=design):return design(a,b,q)@c
  fs.append(f)
 return lambda x:np.column_stack([f(x[:,i],x[:,(i+2)%4]) for i,f in enumerate(fs)])

def calibrate(f):
 raw=s['y_obs_raw'].reshape(-1,4)
 return np.array([root(lambda x:f(x[None,:])[0]-y,np.zeros(4),tol=1e-11).x for y in raw]).ravel()

orig=m.axis_operator
def per(k,v,h,loss):
 if not np.isnan(loss[0]):return orig(k,v,h,loss)
 n=len(k); i=np.arange(n); l=(i-1)%n; r=(i+1)%n; a=-(k+k[l])/(2*h*h)-np.maximum(v,0)/h; b=-(k+k[r])/(2*h*h)-np.maximum(-v,0)/h
 return m.sparse.csc_matrix((np.r_[a,-a-b,b],(np.tile(i,3),np.r_[l,i,r])),shape=(n,n))
m.axis_operator=per
if __name__=='__main__':
 for kind in ['pade2','pade4','sep','mix','inter']:
  E=np.empty_like(Y)
  for j in range(5):
   test=np.arange(len(X))%5==j; E[test]=fit(X[~test],Y[~test],kind)(X[test])-Y[test]
  f=fit(X,Y,kind); y=calibrate(f)
  print('CAL',kind,'fit',np.max(np.std(f(X)-Y,axis=0)),'cv',np.sqrt(np.mean(E**2,axis=0)).round(6),'inverse',np.max(abs(f(y.reshape(-1,4)).ravel()-s['y_obs_raw'])),'range',y.min(),y.max());
  vals=[]
  for i,j in itertools.product(range(3),repeat=2):
   a=[s['robin_x'],0,np.nan][i]; b=[s['robin_y'],0,np.nan][j]; S,U=m.chamber_response(s,[a,a,b,b]); w,c=m._inverse(s,S,y); z0=s['Z_nuisance']; e=S@w-y; prof=e-z0@np.linalg.solve(z0.T@z0+s['nuisance_ridge']*np.eye(z0.shape[1]),z0.T@e); vals.append((np.linalg.norm(prof),i,j,c['primal_objective'],w.round(4)))
  for vv in sorted(vals)[:5]:print('wall',vv)
