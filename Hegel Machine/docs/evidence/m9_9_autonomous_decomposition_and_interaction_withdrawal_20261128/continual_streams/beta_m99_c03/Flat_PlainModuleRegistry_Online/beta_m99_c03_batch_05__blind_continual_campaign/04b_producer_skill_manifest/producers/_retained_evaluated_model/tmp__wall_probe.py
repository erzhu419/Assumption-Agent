import sys,pickle,itertools
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import splu
from _verified_upstream import calibrate_upstream
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=calibrate_upstream(dict(item['study'],reference_tracer_pulses=item['campaign_diagnostics']))
p=s['reference_tracer_pulses']; W=np.stack([a['w_pulse'] for a in p],axis=1); Y=np.stack([a['y_pulse_raw'] for a in p],axis=1)
nx,ny=s['nx'],s['ny']; reaction=sp.diags(s['V_reaction_2d'].ravel(order='F')); I=sp.eye(nx*ny,format='csc'); Q=np.stack([m.ravel(order='F') for m in s['source_modes']],axis=1); sample=s['sensor_ij'] @ [1,nx]
def axis(a,kind):
 k,v,r,h=(s['kappa_'+a],s['vel_'+a],s['robin_'+a],s['L'+a]/(s['n'+a]-1)); n=len(k); i=np.arange(n); prev=(i-1)%n; nxt=(i+1)%n
 l=(k+k[prev])/(2*h*h)+np.maximum(v,0)/h; u=(k+k[nxt])/(2*h*h)+np.maximum(-v,0)/h
 A=np.diag(l+u); A[i,prev]=-l; A[i,nxt]=-u
 if kind=='periodic': return A
 for j,b,d in [(0,1,u[0]),(n-1,n-2,l[-1])]:
  A[j]=0
  if kind in ('robin','neumann','face','double','face0','cap','clamp'):
   c={'robin':k[j]/h**2,'neumann':k[j]/h**2,'face':d,'double':2*k[j]/h**2,'face0':d,'cap':0,'clamp':0}[kind]; A[j,b]=-c; A[j,j]=c+(0 if kind in ('neumann','face0') else r/h)
   if kind=='clamp': A[j,j]+=k[j]/h**2
 return A
def forward(A,B,R=Q):
 L=sp.kron(sp.eye(ny),sp.csc_matrix(A),format='csc')+sp.kron(sp.csc_matrix(B),sp.eye(nx),format='csc')+reaction
 return np.vstack([splu(L+t*I).solve(R)[sample] for t in s['mu_shifts']])
if __name__=='__main__':
 kinds='robin periodic neumann face double face0 cap clamp'.split(); scores=[]
 for x,y in itertools.product(kinds,repeat=2):
  S=forward(axis('x',x),axis('y',y)); e=S@W-Y; scores.append((np.sqrt(np.mean(e**2)),x,y,np.sqrt(np.mean(e.reshape(3,-1,e.shape[-1])**2,axis=1)).round(6).tolist()))
 print('pulse weights',W.T,'data RMS',np.sqrt(np.mean(Y**2))); print('best orderings (rms, x, y, shift-by-pulse rms):'); print(*sorted(scores)[:10],sep='\n'); print('baseline:',next(r for r in scores if r[1:3]==('robin','robin')))
