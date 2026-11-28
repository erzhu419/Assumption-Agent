import importlib.util,pickle,numpy as np,json
from scipy.sparse import diags,eye,kron,coo_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('m','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=p['study']; ds=p['campaign_diagnostics']; w=np.stack([d['w_pulse'] for d in ds],axis=1); y=np.stack([m._calibrate(s,d['y_pulse_raw']) for d in ds],axis=1)
nx,ny=s['nx'],s['ny']; N=nx*ny; ids=np.arange(N); ix,iy=ids%nx,ids//nx; ind=s['sensor_ij']@[1,nx]; B=np.stack([v.ravel(order='F') for v in s['source_modes']],axis=1)@w
L=(kron(eye(ny),m._axis(s['kappa_x'],s['vel_x'],s['Lx']/(nx-1)))+kron(m._axis(s['kappa_y'],s['vel_y'],s['Ly']/(ny-1)),eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc(); I=eye(N,format='csc'); walls=[(ids[ix==0],1),(ids[ix==nx-1],-1),(ids[iy==0],nx),(ids[iy==ny-1],-nx)]; M=[]
for rows,off in walls:
 a=coo_matrix((np.ones(len(rows)),(rows,rows)),shape=(N,N)).tocsc(); b=coo_matrix((np.ones(len(rows)),(rows,rows+off)),shape=(N,N)).tocsc(); M.extend([a,a-b]);
def pred(A): return np.concatenate([splu(A+mu*I).solve(B)[ind] for mu in s['mu_shifts']])
def op(t,Ms): return L+sum((v*a for v,a in zip(t,Ms)),0)
pr=lambda r:m._profile(s,r)
print('baseline by pulse',np.sqrt(np.mean(pr(pred(L)-y)**2,axis=0)))
for mode,Ms in [('diag+normal gradient',M),('normal gradient',M[1::2])]:
 for hold in [-1,0,1,2]:
  keep=np.arange(3)!=hold; fit=least_squares(lambda t:pr(pred(op(t,Ms))-y)[:,keep].ravel(),np.zeros(len(Ms)),gtol=1e-12); r=pr(pred(op(fit.x,Ms))-y); print(mode,hold,fit.x.round(7),np.sqrt(np.mean(r*r,axis=0)), 'jac sv',np.linalg.svd(fit.jac,compute_uv=False)[-1])
for a,n in zip('xy',[nx,ny]):
 h=s['L'+a]/(n-1); k=s['kappa_'+a]; v=s['vel_'+a]; print(a,'h',h,'boundary kappas/h2',k[[0,-1]]/h**2,'face kappas/h2',(k[[0,-1]]+k[[1,-2]])/(2*h*h),'velocity/h',v[[0,-1]]/h,'robin/h',s['robin_'+a]/h)
for f in ['archive_apparatus_log.json','pre_llm_gate_diagnostic.json']:
 try:
  t=json.load(open('/tmp/hegel_continual_archive/'+f)); print(f, ([(r.get('completed_campaign_short_name'),list(r)) for r in t][-2:] if isinstance(t,list) else str(t)[:800]))
 except FileNotFoundError: pass
