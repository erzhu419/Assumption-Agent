import pickle, numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']['sensor_calibration_sweeps']; x,y=s['ref_rosette'],s['raw_rosette']; good=np.max(abs(y),axis=1)<5; x,y=x[good],y[good]; u=x[:,[2,3,0,1]]
funcs={'sin':lambda x,k:np.sin(k*x),'xcos':lambda x,k:x*np.cos(k*x),'exp':lambda x,k:x*np.exp(-k*x*x),'erf':lambda x,k:erf(k*x),'tanh':lambda x,k:np.tanh(k*x),'sin2':lambda x,k:np.sin(k*x*x),'rat':lambda x,k:x/(1+k*x*x)}

def fit(rows,key,extra,ck=2.,shared=False):
 a,b,z=x[rows],u[rows],y[rows]
 f=funcs[key[0]]
 def design(q,a,b):
  return np.stack([np.ones_like(a),a,b,1-np.cos(q[-1]*b),f(a,q[0])]+([a**3] if extra else [])+([funcs[key[1]](a,q[1])] if len(key)==2 else []),axis=-1)
 def coeff(q):
  d=design(q,a,b); return np.array([np.linalg.lstsq(d[:,j],z[:,j],rcond=None)[0] for j in range(4)])
 def cost(q):return (np.einsum('njd,jd->nj',design(q,a,b),coeff(q))-z).ravel()
 starts=[[v,ck] if len(key)==1 else [v,w,ck] for v in (.55,1.4,2.5) for w in ((.5,1.8) if len(key)>1 else (0,))]
 opts=[least_squares(cost,q,bounds=(.04,6.),max_nfev=150,gtol=3e-13) for q in starts]; r=min(opts,key=lambda p:np.linalg.norm(p.fun)); c=coeff(r.x)
 return np.sqrt(np.mean(r.fun**2)),r.x,c,lambda a,b: np.einsum('njd,jd->nj',design(r.x,a,b),c)

rows=np.arange(len(x)); ranking=[]
for k in [('sin',),('xcos',),('exp',),('erf',),('tanh',),('rat',),('sin','exp'),('sin','xcos'),('sin','erf'),('sin','tanh'),('exp','erf'),('exp','xcos')]:
 for e in (False,True):
  v,q,c,p=fit(rows,k,e); ranking.append((v,k,e)); print(k,e,'rmse',v,'scales',q,'channel0',c[0],flush=True)
for v,k,e in sorted(ranking)[:4]:
 er=[]
 for block in np.array_split(rows[np.argsort(np.max(abs(x),axis=1))],6):
  w,q,c,p=fit(np.setdiff1d(rows,block),k,e); er.append(np.max(abs(p(x[block],u[block])-y[block])))
 print('blocked residual max',k,e,er)

# A parity-constrained, shared Chebyshev form avoids poorly determined even diagonal terms.
for deg in (9,11,13,15,17):
 from numpy.polynomial.chebyshev import chebvander
 out=[]; norm=2.
 for block in np.array_split(rows[np.argsort(np.max(abs(x),axis=1))],6):
  a=np.setdiff1d(rows,block); pred=[]
  for j in range(4):
   A=np.column_stack((np.ones(len(x)),chebvander(x[:,j]/norm,deg)[:,1::2],u[:,j],1-np.cos(2*u[:,j])))
   pred.append(A[block]@np.linalg.lstsq(A[a],y[a,j],rcond=None)[0]-y[block,j])
  out.append(np.max(abs(pred)))
 print('parity poly blocked residual max',deg,out)
