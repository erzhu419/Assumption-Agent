import pickle,runpy,numpy as np
from itertools import combinations_with_replacement as cwr
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; m=runpy.run_path('/holdout_task/batch_solver.py'); fit=m['_ls']; aff=m['_affine']; t=s['train']; y=t['y']; tr=t['traj']; x=t['x']; z=m['_calibrate'](s['panel_a'],x); w=m['_panel_b'](s['panel_b'],z); u=s['panel_c']['u']; v1=s['panel_c']['v1']; v2=s['panel_c']['v2']; ix=np.arange(len(y)); a=aff(w); d=cdist(w,w,'sqeuclidean'); splits=[ix[52:],*np.array_split(np.random.default_rng(17).permutation(len(y)),4),np.argsort(np.linalg.norm(w-w.mean(0),axis=1))[-16:],np.argsort(np.partition(d,1,axis=1)[:,1])[-16:]]
def quad(x):return np.c_[aff(x),*[x[:,i]*x[:,j] for i,j in cwr(range(4),2)]]
def profile(v):
 def mat(x,f):return aff(np.tanh(x*f))
 def res(p):
  a=mat(u,p); return (a@fit(a,v)-v).ravel()
 p=least_squares(res,[1.],bounds=(.1,8.)).x; b=fit(mat(u,p),v); print('C1 input tanh',p,np.std(res(p)))
 return lambda x:mat(x,p)@b
c1=profile(v1); c2=lambda x:quad(x)@fit(quad(u),v2)
# Also test output saturation of a linear projection.
betas=[]
for v in v1.T:
 b0=fit(aff(u),v); p=least_squares(lambda p:p[0]+p[1]*np.tanh(aff(u)@p[2:])-v,np.r_[0.,max(abs(v))*1.2,b0/max(abs(v))],max_nfev=300).x; betas.append(p)
def c1out(x):return np.column_stack([p[0]+p[1]*np.tanh(aff(x)@p[2:]) for p in betas])
print('C1 output tanh',np.std(c1out(u)-v1),'C2 quadratic',np.std(c2(u)-v2))
def score(a,k=None):
 er=[]
 for iv in splits:
  it=np.setdiff1d(ix,iv); aa=a[it]
  if k is None:p=a[iv]@fit(aa,y[it])
  else:
   q=np.linalg.solve(k[np.ix_(it,it)]+.005*np.eye(len(it)),np.c_[aa,y[it]]); c=fit(aa.T@q[:,:-1],aa.T@q[:,-1]); p=a[iv]@c+k[np.ix_(iv,it)]@(q[:,-1]-q[:,:-1]@c)
  er.append(np.abs(p-y[iv]).mean())
 return np.round(er,5)
g=np.exp(-.5*d)
for key,base in [('Z',z),('W',w)]:
 for nm,op in [('C1in',c1),('C1out',c1out),('C2',c2)]:
  f=op(base); print(key,nm,'standalone',score(aff(f)),'plusW',score(np.c_[a,f]),'GPplusW',score(np.c_[a,f],g))
print('trajectory coefficient by time',np.round(fit(a,tr).T,5))
