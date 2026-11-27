import pickle
import numpy as np
from scipy.sparse import csr_matrix, eye, kron, diags
from scipy.sparse.linalg import splu
c=pickle.load(open('/holdout_task/public_beta_m96_w03_s00.pkl','rb'))[0]
s,g=c['study'],c['gt']; V=s['V_reaction_2d']; nx,ny=V.shape
vx=V[:,0]; vy=V[0,:]-V[0,0]
print('V additive defect',np.max(np.abs(V-vx[:,None]-vy)), 'V versus reaction_x+reaction_y',np.max(np.abs(V-s['reaction_x'][:,None]-s['reaction_y'])), 'max circulation',max(s['past_vortex_circulation_trace']))
Z=s['Z_nuisance']; G=Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]); i,j=s['sensor_ij'].T
src=s['source_modes'].reshape(-1,nx*ny).T

def build(axis,mode,edge):
 L,n,k,v,r=(s['L'+axis],s['n'+axis],s['kappa_'+axis],s['vel_'+axis],s['robin_'+axis]); h=L/n
 t=2*k[:-1]*k[1:]/(k[:-1]+k[1:])/h**2
 a=np.diag(np.r_[0.,t]+np.r_[t,0.])-np.diag(t,1)-np.diag(t,-1)
 for q in (0,n-1): a[q,q]+=r*k[q]/(h*(r*h/2+k[q]))
 if mode=='none': return csr_matrix(a)
 if mode in ('cell','column'):
  for q,z in enumerate(v/h):
   if z==0: continue
   p=q-(1 if z>0 else -1) if mode=='cell' else q+(1 if z>0 else -1)
   if 0<=p<n:
    a[q,q]+=abs(z)
    if mode=='cell': a[q,p]-=abs(z)
    else: a[p,q]-=abs(z)
   elif edge: a[q,q]+=abs(z)
 else:
  f={'average':(v[:-1]+v[1:])/2,'left':v[:-1],'right':v[1:]}[mode]/h
  for q,z in enumerate(f):
   p=q if z>=0 else q+1
   a[q,p]+=z; a[q+1,p]-=z
  if edge:
   a[0,0]+=max(0,-v[0])/h; a[-1,-1]+=max(0,v[-1])/h
 return csr_matrix(a)

scores=[]
for label,reaction in [('V',V),('components',s['reaction_x'][:,None]+s['reaction_y'])]:
 for mode in ('cell','column','average','left','right','none'):
  for edge in (False,True):
   A=kron(build('x',mode,edge),eye(ny))+kron(eye(nx),build('y',mode,edge))+diags(reaction.ravel())
   fields=np.array([splu((A+mu*eye(nx*ny)).tocsc()).solve(src).T.reshape(-1,nx,ny) for mu in s['mu_shifts']])
   ferr=np.linalg.norm(np.einsum('m,mij->ij',g['w_opt'],fields[-1])-g['terminal_field_A'])/np.linalg.norm(g['terminal_field_A'])
   samples=fields[:,:,i,j]
   for order,a in [('shift_first',samples.transpose(0,2,1)),('sensor_first',samples.transpose(2,0,1))]:
    a=a.reshape(-1,src.shape[1]); a=a-Z@np.linalg.solve(G,Z.T@a)
    scores.append((np.max(np.abs(a-g['S_tilde'])),ferr,label,mode,edge,order))
for row in sorted(scores)[:12]: print(row)
