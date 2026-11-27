import sys,pickle,json,itertools,importlib.util
import numpy as np
from scipy.sparse import diags,eye,kron
from scipy.sparse.linalg import splu
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream

def load(name):
    p='/tmp/hegel_continual_archive/beta_m99_c02_'+name+'_solver.py'
    spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

base,cyclic,latest=map(load,['s00','batch_02','batch_05'])
s0=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
pay=json.load(open('/holdout_task/_upstream_param_payload.json'))
s=calibrate_upstream(s0,pay)
for st in (s0,s):
    sweeps=st['sensor_calibration_sweeps']; r=sweeps['ref_rosette']; o=sweeps['raw_rosette']; ok=(abs(o)<=8.5).all(1)
    b=np.linalg.lstsq(np.c_[np.ones(len(r)),r][ok],o[ok],rcond=None)[0]
    print('sweep inliers',ok.sum(),'affine',np.round(b,7),'rmse',np.sqrt(np.mean((np.c_[np.ones(len(r)),r][ok]@b-o[ok])**2,axis=0)))
axes=[]
for a in 'xy':
    k,v=s['kappa_'+a],s['vel_'+a]; h=s['L'+a]/(len(k)-1); r=s['robin_'+a]
    opts={'R':base._axis(k,v,h,r),'N':base._axis(k,v,h,0),'C':cyclic._axis(k,v,h),'P':latest._axis(k,v,h,True),'F':latest._axis(k,v,h,False)}
    unique={}
    for label,A in opts.items():
        alias=next((key for key,B in unique.items() if np.max(abs((A-B).toarray()))<1e-12),None)
        if alias: print(a,label,'identical to',alias)
        else: unique[label]=A
    axes.append(unique)

y=s['y_obs_raw']; z=s['Z_nuisance']; nr=s['nuisance_ridge']; sr=s['source_ridge']; cap=s['u_box']; total=s['total_budget']; m=len(s['source_modes']); n=len(s['sensor_ij'])

def assess(S, group_id):
    scores=[]
    for group in np.unique(group_id):
        test=group_id==group; train=~test; Z=z[train]; X=S[train]; Y=y[train]
        ridge=Z.T@Z+nr*np.eye(Z.shape[1]); B=np.linalg.solve(ridge,Z.T@np.column_stack([X,Y])); C=np.column_stack([X,Y])-Z@B
        x,t=C[:,:m],C[:,m]; w,_=latest._box_simplex(x.T@x+sr*np.eye(m),-x.T@t,cap,total)
        pred=S[test]@w+z[test]@(B[:,:m]@(-w)+B[:,m]); scores.append(np.sqrt(np.mean((pred-y[test])**2)))
    return np.sqrt(np.mean(np.square(scores))),max(scores)

rows=[]
for (ix,ax),(iy,ay) in itertools.product(*[d.items() for d in axes]):
    S,U=base._responses(s,(ax,ay)); x,t=latest._profile(S,y,s); w,_=latest._box_simplex(x.T@x+sr*np.eye(m),-x.T@t,cap,total)
    loss=np.sqrt(np.mean((x@w-t)**2)); rosette=assess(S,np.tile(np.arange(n)//4,len(s['mu_shifts']))); shift=assess(S,np.repeat(np.arange(len(s['mu_shifts'])),n)); bank=assess(S,s['row_bank_id']); rows.append((rosette[0],ix+iy,loss,rosette[1],*shift,*bank,np.round(w,4)))
print('LO-rosette pooled, walls x/y, fit RMS, worst rosette, LO-shift pooled/worst, LO-bank pooled/worst, w')
for row in sorted(rows): print(row)
