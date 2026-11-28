import pickle, numpy as np, importlib.util
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/beta_m100_c02_s00_solver.py'); old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];s=i['study'];c=s['sensor_calibration_sweeps'];x=c['ref_rosette'];y=c['raw_rosette'];p=i['campaign_diagnostics']; w=np.array([q['w_pulse'] for q in p]).T; raw=np.array([q['y_pulse_raw'] for q in p]);
A=np.column_stack((np.ones(len(x)),x));coef=np.linalg.lstsq(A,y,rcond=None)[0];print('affine',coef,'rms',np.sqrt(np.mean((A@coef-y)**2,axis=0)),'max',np.max(np.abs(A@coef-y),axis=0)); print('bank fields', list(c['single_bank_sweeps'])); print('aux head',s['aux_manifold_telemetry'][:5]);print('sensor',s['sensor_ij'].tolist());
for power in [2,3]:
 B=np.column_stack([A]+[x**j for j in range(2,power+1)]); cr=np.linalg.lstsq(B,y,rcond=None)[0];print('power',power,'rms',np.sqrt(np.mean((B@cr-y)**2,axis=0)), 'extra terms',cr[5:].round(6))
obs=np.linalg.solve(coef[1:].T,(raw.reshape(-1,4)-coef[0]).T).T.reshape(raw.shape).T
nx,ny=s['nx'],s['ny']; h=[s['Lx']/(nx-1),s['Ly']/(ny-1)]; ax,ay=[old.axis_operator(s['kappa_'+a],s['vel_'+a],d,s['robin_'+a]).toarray() for a,d in zip('xy',h)]; ij=s['sensor_ij']; ii=ij[:,0]+nx*ij[:,1]; rhs=s['source_modes'].reshape(-1,nx*ny,order='F').T@w;eye=sparse.eye(nx*ny,format='csc'); reaction=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc');
def predict(theta,kind):
 mats=[ax.copy(),ay.copy()]; npar=len(kind)
 for a,m in enumerate(mats):
  for b,(j,k) in enumerate([(0,1),(-1,-2)]):
   for p,z in zip(kind,theta[(2*a+b)*npar:(2*a+b+1)*npar]):
    if p=='r':m[j,j]+=z
    if p=='d':m[j,j]+=z;m[j,k]-=z
    if p=='o':m[j,j]+=z;m[j,-1 if j==0 else 0]-=z
    if p=='t':m[j,j]+=z;m[j,2 if j==0 else -3]-=z
 L=sparse.kron(sparse.eye(ny),mats[0],format='csc')+sparse.kron(mats[1],sparse.eye(nx),format='csc')+reaction
 return np.concatenate([splu(L+mu*eye).solve(rhs)[ii] for mu in s['mu_shifts']])
def rms(v):return np.sqrt(np.mean(v**2,axis=0))
print('open RMS',rms(predict([], '')-obs));print('old mapping agrees',np.max(abs(predict([], '')-old.chamber_response(s)[0]@w)))
for kind in ['r','d','rd','ro','rdo','rdt','rdot']:
 def fit(ids):return least_squares(lambda t:(predict(t,kind)[:,ids]-obs[:,ids]).ravel(),np.zeros(4*len(kind)),diff_step=1e-5,max_nfev=100,gtol=1e-11,ftol=1e-11,xtol=1e-11)
 f=fit([0,1,2]); cv=[float(rms(predict(fit([j for j in range(3) if j!=k]).x,kind)[:,k]-obs[:,k])) for k in range(3)];print(kind,'params',f.x.reshape(4,-1).round(6).tolist(),'fit',rms(predict(f.x,kind)-obs),'LOPO',cv)
