import numpy as np
from scipy.sparse import diags,eye,kron,csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
import pickle,itertools

def axis(k,v,L,grid,mix,adv):
 n=len(k); h=L/(n-grid); f=(k[:-1]+k[1:])/2 if mix==0 else 2*k[:-1]*k[1:]/(k[:-1]+k[1:]); a=f/h**2
 vf=(v[:-1]+v[1:])/2; p=np.maximum(vf,0)/h; m=np.maximum(-vf,0)/h
 if adv==0: l=-a-p; u=-a-m; d=np.r_[a+p,0]+np.r_[0,a+m]
 elif adv==1: l=-a-np.maximum(v[1:],0)/h; u=-a-np.maximum(-v[:-1],0)/h; d=np.r_[-u,0]+np.r_[0,-l]
 elif adv==2: l=-a-p; u=-a-m; d=np.r_[-u,0]+np.r_[0,-l]
 elif adv==3: l=-a-np.maximum(v[:-1],0)/h; u=-a-np.maximum(-v[1:],0)/h; d=np.r_[-l,0]+np.r_[0,-u]
 else: l=-a-np.maximum(v[1:],0)/h; u=-a-np.maximum(-v[:-1],0)/h; d=np.r_[a,0]+np.r_[0,a]+abs(v)/h
 return diags([l,d,u],[-1,0,1],format='csc')

def build(s,grid,mix,adv):
 nx,ny=s['nx'],s['ny']; ax=axis(s['kappa_x'],s['vel_x'],s['Lx'],grid,mix,adv); ay=axis(s['kappa_y'],s['vel_y'],s['Ly'],grid,mix,adv)
 return csc_matrix(kron(eye(ny),ax)+kron(ay,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F')))

def setup(s):
 nx,ny=s['nx'],s['ny']; ii,jj=np.indices((nx,ny)); edge=np.vstack([a.ravel(order='F') for a in (ii==0,ii==nx-1,jj==0,jj==ny-1)]).T.astype(float)
 idx=s['sensor_ij'][:,0]+nx*s['sensor_ij'][:,1]; return edge,idx

def pred(A,edge,idx,mu,rhs,b):
 D=diags(edge@b); I=eye(A.shape[0],format='csc'); return np.concatenate([splu(A+D+t*I).solve(rhs)[idx] for t in mu])

if __name__=='__main__':
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; edge,idx=setup(s); mu=s['mu_shifts']; pulses=s['reference_tracer_pulses']; W=np.column_stack([p['w_pulse'] for p in pulses]); B=s['source_modes'].reshape(len(W),-1,order='F').T@W; Y=np.column_stack([p['y_pulse_raw'] for p in pulses]); rows=[]
 for grid,mix,adv in itertools.product(range(2),range(2),range(5)):
  A=build(s,grid,mix,adv)
  for symmetry in [2,4]:
   E=edge@np.array([[1,0],[1,0],[0,1],[0,1]]) if symmetry==2 else edge
   def fit(ids, mask):
    return least_squares(lambda b:(pred(A,E,idx,mu,B[:,ids].copy(),b)-Y[:,ids])[mask].ravel(),np.ones(symmetry)*5,bounds=(-0.4,100),max_nfev=45,gtol=1e-10).x
   b=fit([0,1],slice(None)); er=pred(A,E,idx,mu,B,b)-Y
   rows.append((np.sqrt(np.mean(er**2)),grid,mix,adv,symmetry,b,np.max(abs(er))))
 for r in sorted(rows,key=lambda a:a[0])[:20]: print(r)
 for r in sorted(rows,key=lambda a:a[0])[:8]:
  _,grid,mix,adv,sy,b,_=r; A=build(s,grid,mix,adv); E=edge@np.array([[1,0],[1,0],[0,1],[0,1]]) if sy==2 else edge
  cv=[]
  for p in range(2):
   tr=1-p
   q=least_squares(lambda z:pred(A,E,idx,mu,B[:,tr:tr+1].copy(),z).ravel()-Y[:,tr],b,bounds=(-.4,100),max_nfev=40).x
   e=pred(A,E,idx,mu,B[:,p:p+1].copy(),q).ravel()-Y[:,p]; cv.append((np.sqrt(np.mean(e**2)),np.max(abs(e)),q.round(6)))
  print('LOPO',grid,mix,adv,sy,cv)
