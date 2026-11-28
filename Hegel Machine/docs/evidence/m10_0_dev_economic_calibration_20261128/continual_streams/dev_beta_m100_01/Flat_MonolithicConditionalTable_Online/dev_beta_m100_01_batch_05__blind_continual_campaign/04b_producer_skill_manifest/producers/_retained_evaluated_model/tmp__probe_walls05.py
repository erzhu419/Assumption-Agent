import itertools,sys,pickle,numpy as np
from scipy import sparse
from scipy.optimize import root, least_squares
from probe_rational05 import s,X,Y,B,fit,model
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_beta_m100_01_batch_04_solver as m
import dev_beta_m100_01_batch_03_solver as r
import dev_beta_m100_01_batch_02_solver as p

def calibration(kind):
 qs=[]
 for j in range(4):
  take=B['bank_id']==j; xb=np.zeros((take.sum(),4));xb[:,0]=B['ref_signal'][take];order=(j+np.arange(4))%4;x=np.r_[X[:,order],xb];v=np.r_[Y[:,j],B['raw_readout'][take]];ok=abs(v)<5;q=fit(x[ok],v[ok],kind);qs.append(q)
  if kind==6:
   a,b,c,d=x.T;e=model(x[ok],q,kind)-v[ok]; il=np.argsort(abs(e))[-5:]; print('cal',j,np.round(q,4),'high residual',np.round(np.column_stack([x[ok],e])[il],4))
 def pred(t):return np.array([model(t[None,(j+np.arange(4))%4],q,kind)[0] for j,q in enumerate(qs)])
 def inv(y):
  res=root(lambda t:pred(t)-y,y/1.6); assert np.max(abs(res.fun))<1e-7; return res.x
 return np.array([inv(y) for y in s['y_obs_raw'].reshape(-1,4)]).ravel()

def forward(params):
 def axis(a,v):
  args=(s['kappa_'+a],s['vel_'+a],s['L'+a]/(s['n'+a]-1));return p._axis(*args) if v=='P' else r._axis(*args,s['robin_'+a]*v)
 nx,ny=s['nx'],s['ny'];L=sparse.kron(sparse.eye(ny),axis('x',params[0]),format='csc')+sparse.kron(axis('y',params[1]),sparse.eye(nx),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc');return m._forward(s,L)

Z=s['Z_nuisance'];rho=s['nuisance_ridge']; ix=np.arange(len(Z)).reshape(len(s['mu_shifts']),-1,4)
def errors(S,y,train,test):
 z=Z[train];D=np.column_stack((S[train],y[train]));P=np.linalg.solve(z.T@z+rho*np.eye(z.shape[1]),z.T@D);T=D-z@P;w,_=m._qp(T[:,:-1],T[:,-1],s['source_ridge'],s['u_box'],s['total_budget']);return S[test]@w+Z[test]@(P[:,-1]-P[:,:-1]@w)-y[test],w

def cv(S,y,splits):
 v=[np.mean(errors(S,y,np.setdiff1d(ix,part),part.ravel())[0]**2) for part in splits];return np.sqrt([np.mean(v),max(v)])

if __name__=='__main__':
 for kind in (6,11,3):
  y=calibration(kind);np.save('/tmp/batch05_y'+str(kind),y);print('calibration',kind)
  for params in itertools.product([0,1,'P'],repeat=2):
   S,U=forward(params); print(params,np.round(cv(S,y,np.array_split(ix,8,axis=1)),6),np.round(cv(S,y,ix),6),np.round(np.std(errors(S,y,ix.ravel(),ix.ravel())[0]),6))
  for fixed in ((None,None),('P',None),(None,'P')):
   fill=lambda z:tuple(next(z) if a is None else a for a in fixed)
   fun=lambda z:errors(forward(fill(iter(z)))[0],y,ix.ravel(),ix.ravel())[0];q=least_squares(fun,np.ones(fixed.count(None)),bounds=(0,20),diff_step=1e-4,max_nfev=45); print('continuous',fill(iter(q.x)),np.sqrt(np.mean(q.fun**2)))
