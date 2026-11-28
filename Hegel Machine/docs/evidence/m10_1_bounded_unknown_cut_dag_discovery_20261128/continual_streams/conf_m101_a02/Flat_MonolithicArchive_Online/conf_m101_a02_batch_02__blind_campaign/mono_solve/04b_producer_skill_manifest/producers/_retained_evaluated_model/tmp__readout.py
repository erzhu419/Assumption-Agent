import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.linalg import null_space
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,t=[s['train'][k] for k in ('x','y','traj')]; n=len(x)
def aff(u):return np.c_[np.ones(len(u)),u]
def fit(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def poly(z,d):return np.column_stack([np.prod(z[:,i],axis=1) if i else np.ones(len(z)) for k in range(d+1) for i in combinations_with_replacement(range(z.shape[1]),k)])
a=s['panel_a']; u,v=a['u'],a['v']; A=aff(u); g=np.ones(len(u),bool)
for i in range(4):
 c=fit(A[g],v[g]); e=np.linalg.norm(v-A@c,axis=1);g=e<4*np.median(e)
a=(x-c[0])/np.diag(c[1:]); b=s['panel_b']; B=fit(aff(b['u']),b['v']); z=aff(a)@B
np.set_printoptions(precision=4,suppress=True)
def metrics(D,target=y):
 c=fit(D[:52],target[:52]); e=D@c-target
 return np.mean(abs(e[:52])),np.mean(abs(e[52:])),np.max(abs(e[52:])),c
from sklearn.model_selection import KFold
from sklearn.linear_model import OrthogonalMatchingPursuit
from scipy.optimize import least_squares
splits=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72))]+list(KFold(5,shuffle=True,random_state=785).split(y))
def cv(D,ridge=0):
 vals=[]
 for i,j in splits:
  if ridge:
   c=np.linalg.solve(D[i].T@D[i]+ridge*np.diag(np.r_[np.zeros(5),np.ones(D.shape[1]-5)]),D[i].T@y[i]);
  else: c=fit(D[i],y[i]);
  vals.append(np.mean(np.abs(D[j]@c-y[j])))
 return np.round([*vals[:2],np.mean(vals[2:]),max(vals[2:])],5)
for name,w in [('a',a),('b',z)]:
 d=poly(w,3); terms=[''.join(map(str,k)) for j in range(4) for k in combinations_with_replacement(range(4),j)];print(name,'coefficients for degrees 2,3:');print(list(zip(terms[5:],np.round(fit(d[:52],y[:52])[5:],4),np.round(fit(d,y)[5:],4))));
 for r in [0,.001,.01,.03,.1,.3,1]:print('cubic ridge',r,cv(d,r))
 for nterm in [2,3,4,5,6,8,10,14,20]:
  vals=[]; names=[]
  for i,j in splits:
   F=d[:,:5]; H=d[:,5:]; H=H-F@fit(F[i],H[i]); Y=y-F@fit(F[i],y[i]); norm=np.linalg.norm(H[i],axis=0); h=H/norm; m=OrthogonalMatchingPursuit(n_nonzero_coefs=nterm,fit_intercept=False).fit(h[i],Y[i]); selected=np.flatnonzero(m.coef_); D=np.c_[F,H[:,selected]]; vals.append(np.mean(abs(D[j]@fit(D[i],y[i])-y[j])));names.append([terms[k+5] for k in selected]);
  print('OMP',nterm,np.round([*vals[:2],np.mean(vals[2:]),max(vals[2:])],5),names[0])
 # Identify quadratic and cubic interaction tensors in the native coordinates.
 for j in [0,1,2,3]:
  e=y-d[:,:5]@fit(d[:,:5],y); print('coordinate',j,'grouped residuals',[(round(float(w[k,j]),3),round(float(e[k]),3)) for k in np.argsort(w[:,j])[::6]])
print('telemetry',s['telemetry_vec'])
