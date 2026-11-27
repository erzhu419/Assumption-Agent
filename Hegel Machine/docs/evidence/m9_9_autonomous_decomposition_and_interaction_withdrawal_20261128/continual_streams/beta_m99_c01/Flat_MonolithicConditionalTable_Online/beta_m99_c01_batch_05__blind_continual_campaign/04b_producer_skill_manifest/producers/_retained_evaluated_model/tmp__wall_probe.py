import importlib.util, json, pickle
from itertools import product
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csc_matrix,diags,eye,kron
from scipy.sparse.linalg import splu
P='/tmp/hegel_continual_archive/'
def load(i):
 spec=importlib.util.spec_from_file_location('m',P+'beta_m99_c01_batch_0'+str(i)+'_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
periodic,cal=load(2),load(4)
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=it['study']; ds=it['campaign_diagnostics']; raw=np.stack([d['y_pulse_raw'] for d in ds]); w=np.stack([d['w_pulse'] for d in ds]); sw=s['sensor_calibration_sweeps']; co=cal._fit_rosette(sw['ref_rosette'],sw['raw_rosette']); nx,ny=s['nx'],s['ny']; I=eye(nx*ny,format='csc'); reaction=diags(s['V_reaction_2d'].ravel(order='F')); modes=s['source_modes'].reshape(len(w[0]),-1,order='F').T; rhs=modes@w.T; i,j=s['sensor_ij'].T; sens=i+nx*j
axes=[]; params=[]
for c,n in [('x',nx),('y',ny)]:
 k=s['kappa_'+c]; v=s['vel_'+c]; h=s['L'+c]/(n-1); a=periodic._axis(k,v,h).toarray(); axes.append(a); params.append([[(-a[l,r],-a[l,z],k[l]/h**2,s['robin_'+c]/h) for l,r,z in [(0,1,n-1),(n-1,n-2,0)]][q] for q in range(2)])
params=np.array(params).reshape(4,4)
def settings(k):
 return np.array([[0,0,1] if x=='Robin' else [0,0,0] if x=='sealed' else [1,1,0] for x in k]).ravel()
def calc(p):
 aa=[]
 for a,rows,q in zip(axes,p.reshape(2,2,3),params.reshape(2,2,4)):
  a=a.copy(); n=len(a)
  for (l,r,z),(alpha,beta,gamma),(near,wrap,base,rob) in zip([(0,1,n-1),(n-1,n-2,0)],rows,q):
   a[l,r]=-(base+alpha*(near-base));a[l,z]=-beta*wrap;a[l,l]=-a[l,r]-a[l,z]+gamma*rob
  aa.append(csc_matrix(a))
 L=(kron(eye(ny),aa[0])+kron(aa[1],eye(nx))+reaction).tocsc()
 y=np.concatenate([splu(L+u*I).solve(rhs)[sens] for u in s['mu_shifts']]).T
 return (cal._rosette_design(y.reshape(-1,4))@co).reshape(raw.shape)
def errors(d): return np.sqrt(np.mean(d*d,axis=1))
out=[]
for k in product(['Robin','sealed','periodic'],repeat=4):
 d=calc(settings(k))-raw; out.append((float(np.max(errors(d))),k,errors(d)))
for row in sorted(out)[:10]: print('discrete',row)
for a in json.load(open(P+'archive_apparatus_log.json')):
 print('telemetry',a['completed_campaign_short_name'],np.round(np.mean(a['facility_telemetry_trace'],axis=0),3))
p0=settings(min(out)[1]);print('physical scales',params)
for ids in [[0,1,2],[1,2],[0,2],[0,1]]:
 opt=least_squares(lambda p:(calc(p)-raw)[ids].ravel(),p0,max_nfev=80,gtol=1e-12)
 print('free',ids,'RMSE',errors(calc(opt.x)-raw),'coef',np.round(opt.x.reshape(4,3),6),'sv',np.round(np.linalg.svd(opt.jac,compute_uv=False),7))
print('calibration residual absolute quantiles',np.quantile(abs(cal._rosette_design(sw['ref_rosette'])@co-sw['raw_rosette']),[.5,.8,.9,.95,1]))
