import itertools,pickle
import numpy as np
from scipy.sparse import csr_matrix,diags,eye,kron
from scipy.sparse.linalg import splu
p=pickle.load(open('/holdout_task/public_beta_m96_w03_s00.pkl','rb'))[0]; s,t=p['study'],p['gt']; nx,ny=s['nx'],s['ny']; V=s['V_reaction_2d']; R=s['reaction_x'][:,None]+s['reaction_y']; Vsep=V.mean(1)[:,None]+V.mean(0)-V.mean(); Z=s['Z_nuisance']; Q,d,_=np.linalg.svd(Z,full_matrices=False); shrink=d*d/(d*d+s['nuisance_ridge']); I=np.eye(len(Z)); profiles={'residual':I-(Q*shrink)@Q.T,'sqrt':I-(Q*(1-np.sqrt(1-shrink)))@Q.T,'OLS':I-Q@Q.T,'none':I}
print('V range / separability / difference to supplied factors:',V.min(),V.max(),np.max(abs(V-Vsep)),np.max(abs(V-R)), 'circulation max',max(s['past_vortex_circulation_trace']))
c=s['sensor_calibration_sweeps']; cal=[]
for b in range(4):
 m=c['bank_id']==b; x=c['ref_signal'][m]; y=c['raw_readout'][m]; a=np.linalg.lstsq(np.c_[x,np.ones(len(x))],y,rcond=None)[0]; cal.append(a); print('bank',b,'count',len(x),'fit',a,'max residual',max(abs(y-a[0]*x-a[1])))
a=np.array(cal)[s['row_bank_id']]; y=(s['y_obs_raw']-a[:,1])/a[:,0]; print('y errors',[(k,np.linalg.norm(P@y-t['r_y'])) for k,P in profiles.items()]); print('target norms',np.linalg.norm(t['S_tilde']),np.linalg.norm(t['r_y']),np.linalg.norm(t['terminal_field_A']))
def op(axis,how,bound):
 k,v=np.array(s['kappa_'+axis]),np.array(s['vel_'+axis]); n=len(k); h=s['L'+axis]/n; rg=s['robin_'+axis]; j=np.arange(n-1); q=2*k[:-1]*k[1:]/(k[:-1]+k[1:])/h**2; A=np.zeros((n,n)); A[j,j]+=q; A[j+1,j+1]+=q; A[j,j+1]-=q; A[j+1,j]-=q; A[0,0]+=rg*k[0]/(h*(rg*h/2+k[0])); A[-1,-1]+=rg*k[-1]/(h*(rg*h/2+k[-1])); vp=np.maximum(v,0)/h; vm=np.maximum(-v,0)/h
 if how=='point':
  A[j+1,j]-=vp[1:]; A[j,j+1]-=vm[:-1]; A[j+1,j+1]+=vp[1:]; A[j,j]+=vm[:-1]
  if bound in ('in','both'): A[0,0]+=vp[0]; A[-1,-1]+=vm[-1]
  if bound=='both': A[0,0]+=vm[0]; A[-1,-1]+=vp[-1]
 else:
  f={'avg':(v[:-1]+v[1:])/2,'left':v[:-1],'right':v[1:],'upstream':np.where(v[:-1]+v[1:]>=0,v[:-1],v[1:])}[how]/h; p=np.maximum(f,0); m=np.maximum(-f,0); A[j,j]+=p; A[j+1,j]-=p; A[j,j+1]-=m; A[j+1,j+1]+=m
  if bound in ('out','both'): A[0,0]+=vm[0]; A[-1,-1]+=vp[-1]
  if bound in ('in','both'): A[0,0]+=vp[0]; A[-1,-1]+=vm[-1]
 return csr_matrix(A)
if __name__=='__main__':
 ij=s['sensor_ij']; src=s['source_modes'].reshape(12,-1).T; ref=t['S_tilde']; norms=np.linalg.norm(ref); shifts=s['mu_shifts']; Id=eye(nx*ny,format='csc'); ranks=[]
 for (how,bound),(r,W) in itertools.product([('point',b) for b in ('none','in','both')]+[(h,b) for h in ('avg','left','right','upstream') for b in ('none','in','out','both')],[('factors',R),('V',V),('Vsep',Vsep),('both',R+V),('factors+mean',R+(V-R).mean())]):
  A=kron(op('x',how,bound),eye(ny))+kron(eye(nx),op('y',how,bound))+diags(W.ravel()); Us=[splu((A+mu*Id).tocsc()).solve(src) for mu in shifts]; F=np.array(Us)[:,ij[:,0]*ny+ij[:,1],:]; field=(Us[-1]@t['w_opt']).reshape(nx,ny); ef=np.linalg.norm(field-t['terminal_field_A'])/np.linalg.norm(t['terminal_field_A']);
  for order,S in [('shift',F.reshape(288,12)),('sensor',F.transpose(1,0,2).reshape(288,12))]:
   for prof,P in profiles.items(): ranks.append((np.linalg.norm(P@S-ref)/norms,ef,how,bound,r,order,prof))
 print('best design/field errors:'); print(*sorted(ranks)[:16],sep='\n'); print('best field:',min(ranks,key=lambda a:a[1]))
