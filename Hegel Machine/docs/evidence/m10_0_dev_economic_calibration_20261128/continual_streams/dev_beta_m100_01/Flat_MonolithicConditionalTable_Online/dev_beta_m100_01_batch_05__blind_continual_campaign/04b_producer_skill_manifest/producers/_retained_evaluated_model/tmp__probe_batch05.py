import sys,pickle,itertools,numpy as np
from scipy import sparse
from scipy.optimize import least_squares,root
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_beta_m100_01_batch_04_solver as m
import dev_beta_m100_01_batch_03_solver as r
import dev_beta_m100_01_batch_02_solver as p
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; sw=s['sensor_calibration_sweeps']; ref,raw=sw['ref_rosette'],sw['raw_rosette']; bank=sw['single_bank_sweeps']; obs=s['y_obs_raw'].reshape(-1,4)
print('sweep ranges',ref.min(0),ref.max(0),[(k,np.shape(v),np.min(v),np.max(v)) for k,v in bank.items()]); print('aux correlations',np.round(np.corrcoef(s['aux_manifold_telemetry'].T),2))
base=m._calibration(sw)
def feat(x,j):return x[:,j], x[:,(j+1)%4], x[:,(j-1)%4], x[:,(j+2)%4]
def pred(x,j,q,kind):
 a,b,c,d=feat(x,j)
 if kind=='poly': return q[0]+q[1]*a+q[2]*b+q[3]*c+q[4]*a*b+q[5]*a**3
 if kind=='poly5': return q[0]+q[1]*a+q[2]*b+q[3]*c+q[4]*a*b+q[5]*a**3+q[6]*a**5
 if kind=='tanh': f=np.tanh(q[5]*a)/q[5]
 elif kind=='sin': f=np.sin(q[5]*a)/q[5]
 elif kind=='atan': f=np.arctan(q[5]*a)/q[5]
 elif kind=='sqrt': f=a/np.sqrt(1+(q[5]*a)**2)
 return q[0]+q[1]*f+q[2]*b+q[3]*c+q[4]*a*b

def fit(kind):
 qs=[]; cvs=[]
 for j in range(4):
  use=bank['bank_id']==j; xb=np.zeros((use.sum(),4));xb[:,j]=bank['ref_signal'][use];X=np.vstack((ref,xb));Y=np.r_[raw[:,j],bank['raw_readout'][use]]
  q0=base[j].copy()
  if kind=='poly5':q0=np.r_[q0,0.]
  elif kind!='poly':q0[5]=0.7
  def estimate(ix):return least_squares(lambda q: pred(X[ix],j,q,kind)-Y[ix],q0,loss='cauchy',f_scale=.002).x
  res=np.zeros(len(Y))
  for k in range(5):
   mask=np.arange(len(Y))%5==k;q=estimate(~mask);res[mask]=pred(X[mask],j,q,kind)-Y[mask]
  cvs.append(np.quantile(abs(res),[.5,.8,.9,.95]));qs.append(estimate(np.ones(len(Y),bool)))
 print(kind,np.round(np.array(cvs),6).tolist(), 'coeff',np.round(qs,4).tolist())
 def inv(y):
  return root(lambda x: np.array([pred(x[None,:],j,qs[j],kind)[0] for j in range(4)])-y,(y-base[:,0])/base[:,1]).x
 return np.array(list(map(inv,obs))).ravel()
ys={k:fit(k) for k in ('poly','poly5','tanh','sin','atan','sqrt')};ys['archive']=m._calibrate(s)
Z=s['Z_nuisance']; ridge=s['nuisance_ridge']; splits=np.array_split(np.arange(len(obs)*4).reshape(3,-1,4),8,axis=1)
for aa,bb in itertools.product('ROP',repeat=2):
 def axis(a,kind):
  args=(s['kappa_'+a],s['vel_'+a],s['L'+a]/(s['n'+a]-1));return {'R':lambda:m._axis(*args),'O':lambda:r._axis(*args,s['robin_'+a]),'P':lambda:p._axis(*args)}[kind]()
 L=sparse.kron(sparse.eye(s['ny']),axis('x',aa),format='csc')+sparse.kron(axis('y',bb),sparse.eye(s['nx']),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc');S,_=m._forward(s,L)
 out=[]
 for name,y in ys.items():
  errs=[]
  for part in [*splits,np.arange(len(y))]:
   test=part.ravel();train=np.setdiff1d(np.arange(len(y)),test) if len(test)<len(y) else test;z=Z[train];D=np.column_stack((S[train],y[train]));P=np.linalg.solve(z.T@z+ridge*np.eye(z.shape[1]),z.T@D);T=D-z@P;w,_=m._qp(T[:,:-1],T[:,-1],s['source_ridge'],s['u_box'],s['total_budget']);e=S[test]@w+Z[test]@(P[:,-1]-P[:,:-1]@w)-y[test];errs.append(np.mean(e*e))
  out.append((name,*np.round(np.sqrt([np.mean(errs[:-1]),max(errs[:-1]),errs[-1]]),6)))
 print(aa+bb,out)
