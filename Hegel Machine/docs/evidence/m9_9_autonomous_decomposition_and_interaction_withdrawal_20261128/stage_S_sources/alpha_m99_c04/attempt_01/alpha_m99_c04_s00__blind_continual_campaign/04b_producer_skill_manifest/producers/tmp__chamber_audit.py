import sys, pickle, json
from pathlib import Path
sys.path.insert(0, '/holdout_task')
import numpy as np
import batch_solver as m
from scipy.optimize import least_squares
D=pickle.load(open('public_batch.pkl','rb'))[0]['dataset']; T=D['train']; X,_=m._features(T,D['test']); b=m._stress_map(D['calibration_load_cells']); dt=D['past_dt']; t=D['eval_times']
A=Path('/tmp/hegel_continual_archive'); print('archive files', sorted(p.name for p in A.iterdir())); p=A/'archive_manifest.json'; print('manifest',p.read_text()[:1800] if p.exists() else None)
N=len(X); labs=T['lab_id']; u=T['aux_chamber_telemetry']; y=np.c_[T['strain_curves_obs']-T['strain_0'][:,None],T['instant_rate_obs']]; K=len(t)
def group(mask): return {k:v[mask] for k,v in T.items()}
def nuisance(B, tau):
    # Pairwise telemetry contrasts fall in the main-effect span. Include explicit quadratic interactions and bounded rational variants.
    U=np.column_stack([u, *[u[:,i]*u[:,j] for i in range(4) for j in range(i,4)], *[(u[:,i]-u[:,j])/(1+(u[:,i]+u[:,j])**2) for i in range(4) for j in range(i+1,4)]])
    temporal=np.stack([B[:,:,0],B[:,:,1],B[:,:,2]],axis=2)
    return (temporal[:,:,:,None]*U[:,None,None,:]).reshape(N,K+1,-1)
errors={key:np.zeros_like(y) for key in ['material','telemetry']}
for lab in np.unique(labs):
    te=labs==lab; tr=~te; tau,w=m._fit(group(tr),X[tr],b,dt,t); C,R=m._design(T,X,b,dt,t,tau); B=np.concatenate([C,R[:,None,:]],axis=1); base=B@w; e=y-base; errors['material'][te]=e[te]
    F=nuisance(B,tau); scales=np.r_[np.repeat(m._mad(e[tr,:K]),K),m._mad(e[tr,K])]; nf=F.shape[-1]; F=F/scales[None,:,None]; Z=F[tr].reshape(-1,nf); s=np.sqrt(np.mean(Z*Z,axis=0)); Z/=s; target=(e[tr]/scales).ravel(); penal=4.
    opt=least_squares(lambda a:np.r_[Z@a-target,penal*a], np.zeros(nf),jac=lambda a:np.vstack([Z,penal*np.eye(nf)]),loss='soft_l1',f_scale=1.5,max_nfev=60)
    errors['telemetry'][te]=e[te]-(F[te]@(opt.x/s))*scales
for label,e in errors.items():
    for name,i in [('full',np.ones(N,dtype=bool)),('missing',np.isnan(T['coupon_features_raw']).any(1)), *[(str(l),labs==l) for l in np.unique(labs)]]:
        print(label,name,'p90 early/mid/late/rate',np.round([np.quantile(np.abs(e[i,j]),.9) for j in [slice(0,4),slice(4,8),slice(8,K),K]],6))
