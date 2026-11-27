import glob,runpy,pickle,numpy as np,itertools
from scipy.sparse import csc_matrix,eye,kron,diags
from scipy.sparse.linalg import splu
m=runpy.run_path(glob.glob('/tmp/hegel_continual_archive/*solver.py')[0]); item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; nx,ny=s['nx'],s['ny']; Y=np.column_stack([m['_calibrate'](dict(s,y_obs_raw=p['y_pulse_raw'])) for p in item['campaign_diagnostics']]); W=np.column_stack([p['w_pulse'] for p in item['campaign_diagnostics']]); F=np.column_stack([r.ravel(order='F') for r in s['source_modes']])@W; ix,iy=s['sensor_ij'].T; I=eye(nx*ny,format='csc'); V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0];
def ax(a,kind):
 k=s['kappa_'+a]; v=s['vel_'+a]; n=len(k); h=s['L'+a]/(n-1); r=s['robin_'+a]; A=m['_axis'](k,v,h,r).toarray();
 for i,j,l,d in [(0,1,n-1,1),(n-1,n-2,0,-1)]:
  if kind=='open': continue
  A[i]=0
  if kind=='dirichlet': A[i,i]=1e7; continue
  inner=k[i]/h**2; wrap=0; leak=0
  if kind=='reflect2': inner*=2
  if kind in ('face','faceadv','periodic','periodicnoadv'): inner=(k[i]+k[j])/(2*h*h)
  if kind in ('periodic','periodicnoadv','periodiclocal'): wrap=(k[i]+k[l])/(2*h*h) if kind!='periodiclocal' else k[i]/h**2
  if kind in ('faceadv','periodic','periodiclocal','localadv'): inner+=max(-d*v[i],0)/h
  if kind in ('periodic','periodiclocal'): wrap+=max(d*v[i],0)/h
  if kind=='half': leak=.5*r/h
  A[i,i]=inner+wrap+leak; A[i,j]=-inner; A[i,l]=-wrap
 return csc_matrix(A)
def predict(A,B,F=F):
 L=kron(eye(ny),A,format='csc')+kron(B,eye(nx),format='csc')+V
 return np.vstack([splu(L+u*I).solve(F)[ix+nx*iy,:] for u in s['mu_shifts']])
if __name__=='__main__':
 ks=('open','reflect','reflect2','face','faceadv','periodic','periodicnoadv','periodiclocal','localadv','half','dirichlet'); ab={a:{k:ax(a,k) for k in ks} for a in 'xy'}; out=[]
 for x,y in itertools.product(ks,repeat=2):
  e=predict(ab['x'][x],ab['y'][y])-Y; ep=e-Q@(Q.T@e); rms=np.sqrt(np.mean(e.reshape(3,-1,2)**2,axis=1)); out.append((np.linalg.norm(e),x,y,rms.tolist(),np.linalg.norm(ep)))
 for row in sorted(out)[:18]: print(row)
