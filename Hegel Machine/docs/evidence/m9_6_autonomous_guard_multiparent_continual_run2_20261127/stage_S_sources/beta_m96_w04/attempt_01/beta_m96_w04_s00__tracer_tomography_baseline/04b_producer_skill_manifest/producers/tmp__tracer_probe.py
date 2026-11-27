import pickle,numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
p=pickle.load(open('/holdout_task/public_beta_m96_w04_s00.pkl','rb'))[0]; s,g=p['study'],p['gt']
def build(axis,mode):
 k=s['kappa_'+axis]; v=s['vel_'+axis]; rg=s['robin_'+axis]; n=len(k); h=s['L'+axis]/n
 a=2*k[:-1]*k[1:]/(k[:-1]+k[1:])/h**2
 A=np.diag(np.r_[a,0]+np.r_[0,a])+np.diag(-a,1)+np.diag(-a,-1)
 for i in (0,n-1): A[i,i]+=rg*k[i]/(h*(rg*.5*h+k[i]))
 if mode.startswith('row'):
  for i,t in enumerate(v):
   j=i-(1 if t>0 else -1)
   if 0<=j<n: A[i,i]+=abs(t)/h; A[i,j]-=abs(t)/h
   elif mode=='row_zero': A[i,i]+=abs(t)/h
 elif mode.startswith('face'):
  for i,t in enumerate((v[:-1]+v[1:])/2):
   j=i if t>0 else i+1; A[i,j]+=t/h; A[i+1,j]-=t/h
  if mode=='face_out': A[0,0]+=max(-v[0],0)/h; A[-1,-1]+=max(v[-1],0)/h
  if mode=='face_copy': A[0,0]-=v[0]/h; A[-1,-1]+=v[-1]/h
 elif mode=='col_zero':
  for i,t in enumerate(v):
   j=i+(1 if t>0 else -1); A[i,i]+=abs(t)/h
   if 0<=j<n: A[j,i]-=abs(t)/h
 return sparse.csc_matrix(A)
Z=s['Z_nuisance']; Q,d,_=np.linalg.svd(Z,full_matrices=False); l=s['nuisance_ridge']; y=s['y_obs_raw']; transforms={'ridge_resid':d*d/(d*d+l),'ridge_sqrt':1-np.sqrt(l/(d*d+l)), 'unregularized':np.ones_like(d)}
for name,t in transforms.items(): print(name,'target error',np.linalg.norm(y-Q@(t*(Q.T@y))-g['r_y']))
V=s['V_reaction_2d']; R=s['reaction_x'][:,None]+s['reaction_y']; nx,ny=V.shape; E=sparse.eye(nx*ny,format='csc'); M=len(s['source_modes']); src=s['source_modes'].reshape(M,-1).T; ij=s['sensor_ij']; obs=ij[:,0]*ny+ij[:,1]; rows=[]
for vx in ('row_zero','row_copy','face_out','face_copy','face_closed','col_zero','none'):
 for vy in ('row_zero','row_copy','face_out','face_copy','face_closed','col_zero','none'):
  base=sparse.kron(build('x',vx),sparse.eye(ny),format='csc')+sparse.kron(sparse.eye(nx),build('y',vy),format='csc')
  for name,react in [('axes',R),('2d',V)]:
   A=base+sparse.diags(react.ravel(),format='csc'); U=[splu(A+mu*E).solve(src) for mu in s['mu_shifts']]; F=(U[-1]@g['w_opt']).reshape(nx,ny)
   ferr=np.linalg.norm(F-g['terminal_field_A'])/np.linalg.norm(g['terminal_field_A']); UU=np.array(U)[:,obs,:]
   for order,S in [('shift',UU.reshape(-1,M)),('sensor',UU.transpose(1,0,2).reshape(-1,M))]:
    for kind,t in transforms.items():
     SS=S-Q@(t[:,None]*(Q.T@S)); serr=np.linalg.norm(SS-g['S_tilde'])/np.linalg.norm(g['S_tilde']); rows.append((ferr+serr,ferr,serr,vx,vy,name,order,kind))
for r in sorted(rows)[:12]: print(r)
