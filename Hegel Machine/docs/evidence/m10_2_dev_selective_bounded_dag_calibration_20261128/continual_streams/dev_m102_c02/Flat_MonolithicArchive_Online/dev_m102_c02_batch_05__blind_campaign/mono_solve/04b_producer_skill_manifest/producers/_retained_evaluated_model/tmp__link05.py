import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from exact05 import D,A,F,folds,y,yt,s,np
from scipy.optimize import least_squares
D=D[:72];a=A(D);p1=D[:,0]*D[:,2];p2=D[:,1]*D[:,3];h=np.c_[a,p1+p2];h2=np.c_[a,p1,p2]
funcs={
 'log':lambda v,k:np.sign(v)*np.log1p(k*abs(v))/k,
 'sqrt':lambda v,k:2*v/(np.sqrt(1+k*abs(v))+1),
 'absrat':lambda v,k:v/(1+k*abs(v)),
 'expsigned':lambda v,k:np.sign(v)*(-np.expm1(-k*abs(v)))/k,
 'atan':lambda v,k:np.arctan(k*v)/k,
 'asinh':lambda v,k:np.arcsinh(k*v)/k,
 'smoothsqrt':lambda v,k:v/np.sqrt(1+k*v*v),
 'signed-sq':lambda v,k:v-k*v*abs(v),
 'sqrtsmooth':lambda v,k:v*2/(np.sqrt(1+k*v*v)+1),
 'exp-damp':lambda v,k:v*np.exp(-k*abs(v)),
 'signedpow':lambda v,k:v/(1+k*abs(v))**.5,
}
for lab,fn in funcs.items():
 for pair in [h,h2]:
  vals=[]; qs=[]; ers=[]
  for i,j in folds:
   def predict(q):return fn(pair@q[:-1],q[-1]);
   q=least_squares(lambda q:predict(q)[i]-y[i],np.r_[F(pair[i],y[i]),.22],bounds=([-np.inf]*pair.shape[1]+[1e-5],[np.inf]*pair.shape[1]+[8]),gtol=1e-11).x
   vals.append(np.mean(abs(predict(q)[j]-y[j]))); qs.append(q); ers.append(np.std(predict(q)[i]-y[i]))
  print(lab,pair.shape[1],np.round([vals[0],np.mean(vals[1:]),np.max(vals[1:]),np.mean(ers)],7),np.round(np.median(qs,axis=0),5))
print('times',s['eval_times']); print('trajectory coefficients (transpose)',F(a,yt).T); print('traj residual PCA',np.linalg.svd(yt-a@F(a,yt),compute_uv=False)); print('rank2 coeff',np.linalg.svd(F(a,yt),compute_uv=False))
