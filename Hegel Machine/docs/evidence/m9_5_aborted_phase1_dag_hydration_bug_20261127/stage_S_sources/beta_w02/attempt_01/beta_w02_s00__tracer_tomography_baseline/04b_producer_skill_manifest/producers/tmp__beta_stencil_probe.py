import sys, pickle, itertools
sys.path.insert(0, '/holdout_task')
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
import beta_w02_s00_solver as m
it = pickle.load(open('/holdout_task/public_beta_w02_s00.pkl','rb'))[0]
s,g = it['study'],it['gt']
nx,ny=s['nx'],s['ny']; M=len(s['source_modes']); f=s['source_modes'].reshape(M,-1).T
I=sparse.eye(nx*ny,format='csc'); i,j=s['sensor_ij'].T; idx=i*ny+j
Z=s['Z_nuisance']; G=Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]); P=lambda a:a-Z@np.linalg.solve(G,Z.T@a)
V=s['V_reaction_2d']; rx,ry=s['reaction_x'],s['reaction_y']; vx=V.mean(1); vy=V.mean(0)-V.mean()
print('separability error',abs(V-vx[:,None]-vy).max())
# Compare combinations of cell upwind, face upwind and boundary flux conventions.
def axis(q,react,grammar,bound):
 n=s['n'+q]; h=s['L'+q]/n; v=s['vel_'+q]/h
 A=m._build_1d(s['L'+q],n,s['kappa_'+q],np.zeros(n),s['robin_'+q],react).toarray()
 l=np.maximum(v[1:],0); u=np.maximum(-v[:-1],0)
 if grammar.startswith('face'):
  w={'avg':(v[:-1]+v[1:])/2,'left':v[:-1],'right':v[1:]}[grammar.split('_')[1]]
  l=np.maximum(w,0); u=np.maximum(-w,0)
 k=np.arange(n-1); A[k+1,k]-=l; A[k,k+1]-=u
 if grammar.endswith('cons'):
  A[k,k]+=l; A[k+1,k+1]+=u
 else:
  A[k+1,k+1]+=l; A[k,k]+=u
 if bound in ('in','both'): A[0,0]+=max(v[0],0); A[-1,-1]+=max(-v[-1],0)
 if bound in ('out','both'): A[0,0]+=max(-v[0],0); A[-1,-1]+=max(v[-1],0)
 return sparse.csc_matrix(A)
options=list(itertools.product(['cell_adv','cell_cons','face_avg_adv','face_avg_cons','face_left_adv','face_left_cons','face_right_adv','face_right_cons'],['none','in','out','both']))
rank=[]; mu=s['mu_shifts'][-1]; target=g['terminal_field_A'].ravel(); src=f@g['w_opt']
for lab,(a,b) in {'V':(vx,vy),'V+reaction':(vx+rx,vy+ry),'reaction':(rx,ry)}.items():
 bx=[sparse.kron(axis('x',a,*o),sparse.eye(ny),format='csc') for o in options]
 by=[sparse.kron(sparse.eye(nx),axis('y',b,*o),format='csc') for o in options]
 for k,X in enumerate(bx):
  for l,Y in enumerate(by):
   A=X+Y; res=np.max(np.abs((A+mu*I)@target-src)); rank.append((res,lab,options[k],options[l],A))
for res,lab,ox,oy,A in sorted(rank,key=lambda r:r[0])[:10]:
 rows=[]
 for mu in s['mu_shifts']:
  u=splu(A+mu*I).solve(f); rows.append(u[idx])
 print(lab,ox,oy,'terminal PDE residual',res,'terminal error',abs(u@g['w_opt']-target).max(),'design error',abs(P(np.vstack(rows))-g['S_tilde']).max())
