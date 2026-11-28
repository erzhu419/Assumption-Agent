import pickle,importlib.util,itertools,numpy as np
from scipy.sparse import diags
p='/tmp/hegel_continual_archive/beta_m100_c03_batch_02_solver.py'; spec=importlib.util.spec_from_file_location('past',p); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=m._calibrate(s,s['y_obs_raw']); Z=s['Z_nuisance'];cap=s['u_box'];budget=s['total_budget'];nz=Z.shape[1]
def axis(d,kind):
 k=s['kappa_'+d];v=s['vel_'+d]; h=s['L'+d]/(len(k)-1); r=s['robin_'+d]; face=(k[:-1]+k[1:])/(2*h*h)
 if kind=='periodic': return m._axis(k,v,h)
 l=np.r_[-face[:-1]-np.maximum(v[1:-1],0)/h,-k[-1]/h**2]; u=np.r_[-k[0]/h**2,-face[1:]-np.maximum(-v[1:-1],0)/h]; b=np.r_[-u[0],-l[:-1]-u[1:],-l[-1]]
 if isinstance(kind,(float,int)): b[[0,-1]]+=kind*r/h
 if kind=='dirichlet': b[[0,-1]]=1;l[-1]=0;u[0]=0
 if kind=='absorbing': b[[0,-1]]*=2
 return diags((l,b,u),(-1,0,1),format='csc')
def evaluate(S, idx, ridge, cap=cap):
 zz=Z[idx]; a=np.c_[S[idx],y[idx]]; q=a-zz@np.linalg.solve(zz.T@zz+s['nuisance_ridge']*np.eye(nz),zz.T@a); X,b=q[:,:-1],q[:,-1]; w,_=m._box_qp(X.T@X+ridge*np.eye(X.shape[1]),-X.T@b,cap,budget); err=y-S@w; gamma=np.linalg.solve(zz.T@zz+s['nuisance_ridge']*np.eye(nz),zz.T@err[idx]); return err-Z@gamma,w
if __name__=='__main__':
 kinds=[0,.25,.5,1,2,4,16,'periodic','absorbing']; n=len(y); i=np.arange(n); Ax={k:axis('x',k) for k in kinds}; Ay={k:axis('y',k) for k in kinds}
 for ridge in [s['source_ridge'],1e-6]:
  results=[]
  for a,b in itertools.product(kinds,repeat=2):
   S,_=m._responses(s,(Ax[a],Ay[b])); e,w=evaluate(S,i,ridge); fs=[]
   for fold in range(6):
    test=(i//4)%6==fold; err,_=evaluate(S,~test,ridge); fs.append(np.sqrt(np.mean(err[test]**2)))
   results.append((float(np.mean(fs)),max(fs),float(np.sqrt(np.mean(e**2))),a,b,w.round(3)))
  print('ridge',ridge); print(*sorted(results,key=lambda x:x[0])[:14],sep='\n'); print('core',*[r for r in results if r[3] in [0,1,'periodic'] and r[4] in [0,1,'periodic']],sep='\n')
