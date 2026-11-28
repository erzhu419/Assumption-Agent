import sys, pickle, json, itertools, importlib.util
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
p='/holdout_task/'
item=pickle.load(open(p+'public_batch.pkl','rb'))[0]; st=item['study']; pay=json.load(open(p+'_upstream_param_payload.json')); ds=item['campaign_diagnostics']
W=np.stack([d['w_pulse'] for d in ds], axis=1)
Y=np.stack([calibrate_upstream(dict(st,y_obs_raw=d['y_pulse_raw']),pay)['y_obs_raw'] for d in ds], axis=1)
Z=st['Z_nuisance']; profile=lambda e: e-Z@np.linalg.lstsq(Z,e,rcond=None)[0]
spec=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/beta_m100_c04_batch_04_solver.py'); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
nx,ny=st['nx'],st['ny']; ij=st['sensor_ij']; rows=ij[:,0]+nx*ij[:,1]; ident=sparse.eye(nx*ny,format='csc'); react=sparse.diags(st['V_reaction_2d'].ravel(order='F'),format='csc'); rhs=st['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T@W

def axis(a, mode):
 k=st['kappa_'+a]; v=st['vel_'+a]; h=st['L'+a]/(len(k)-1); r=st['robin_'+a]; n=len(k)
 if mode=='periodic': return old._axis(k,v,h,True)
 m=old._axis(k,v,h,False).tolil()
 for i,j in [(0,1),(n-1,n-2)]:
  if mode=='open' or (mode=='left_open' and i==0) or (mode=='right_open' and i>0): m[i,i]+=r/h
  if mode=='dirichlet': m[i,i],m[i,j]=1,0
  if mode=='mirror': m[i,i],m[i,j]=2*m[i,i],2*m[i,j]
  if mode=='face_sealed': m[i,i]=(k[i]+k[j])/(2*h*h); m[i,j]=-m[i,i]
  if mode=='upwind_sealed':
   q=max(-v[i] if i==0 else v[i],0)/h; m[i,i]+=q; m[i,j]-=q
 return m.tocsc()

def predict(ax,ay):
 L=sparse.kron(sparse.eye(ny),ax,format='csc')+sparse.kron(ay,sparse.eye(nx),format='csc')+react
 return np.vstack([splu(L+mu*ident).solve(rhs)[rows] for mu in st['mu_shifts']])

if __name__=='__main__':
 modes=['open','sealed','periodic','left_open','right_open','dirichlet','mirror','face_sealed','upwind_sealed']; scores=[]
 for x,y in itertools.product(modes,repeat=2):
  E=predict(axis('x',x),axis('y',y))-Y; s=np.sqrt(np.mean(E**2,axis=0)); scores.append((np.sqrt(np.mean(E**2)),x,y,np.round(s,7),np.sqrt(np.mean(profile(E)**2))))
 for r in sorted(scores,key=lambda r:r[0])[:16]: print(r)
 print('baseline',next(r for r in scores if r[1:3]==('open','open')))
 print('archived',next(r for r in scores if r[1:3]==('periodic','sealed')))
 print('Z std', Z.std(0))
