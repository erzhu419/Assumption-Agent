import importlib.util,json,pickle
from pathlib import Path
import numpy as np
from scipy import sparse
p=Path('/holdout_task'); spec=importlib.util.spec_from_file_location('batch_solver',p/'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
item=pickle.loads((p/'public_batch.pkl').read_bytes())[0]; s=item['study']; cal=s['sensor_calibration_sweeps']; x=cal['ref_rosette']; y=cal['raw_rosette']; n=len(x)
log=Path('/tmp/hegel_continual_archive/archive_apparatus_log.json')
if log.exists():
    for entry in json.loads(log.read_text()):
        print('prefix metadata:',{k:(f'list[{len(v)}]' if isinstance(v,list) else v) for k,v in entry.items()})
print('calibration rows:',n,'diagnostics:',type(item.get('campaign_diagnostics')).__name__,'study fields:',sorted(s))
pairs=[(a,b) for a in range(4) for b in range(a+1,4)]
features={'affine':np.c_[np.ones(n),x], 'quadratic_cross':np.c_[np.ones(n),x,x*x,*[x[:,a]*x[:,b] for a,b in pairs]], 'rational_contrasts':np.c_[np.ones(n),x,x**3,*[(x[:,a]+sign*x[:,b])/(1+rate*(x[:,a]+sign*x[:,b])**2) for a,b in pairs for sign in (-1,1) for rate in (.2,1.)]]}
for name,D in features.items():
    e=np.empty_like(y)
    for test in np.array_split(np.arange(n),5):
        train=np.ones(n,bool); train[test]=False
        e[test]=D[test]@np.linalg.lstsq(D[train],y[train],rcond=None)[0]-y[test]
    print('calibration',name,'OOF channel RMSE',np.sqrt(np.mean(e*e,axis=0)),'worst block-channel RMSE',max(np.sqrt(np.mean(e[test]**2,axis=0)).max() for test in np.array_split(np.arange(n),5)))
obs=m._calibrated_observations(s); Z=s['Z_nuisance']; ij=s['sensor_ij']; group=np.tile(ij,len(s['mu_shifts'])); groups=np.tile(ij,(len(s['mu_shifts']),1)); nr=s['nuisance_ridge']; cap=s['u_box']; budget=s['total_budget']; axis=m._axis_operator
for name in ('robin','face_average_robin','doubled_wall_flux','reflecting','strong_robin'):
    def variant(k,v,r,L):
        A=axis(k,v,r,L).toarray(); h=L/(len(k)-1)
        for i,j in ((0,1),(-1,-2)):
            d=k[i]/h**2; leak=r/h
            if name=='face_average_robin': d=(k[i]+k[j])/(2*h*h)
            if name=='doubled_wall_flux': d*=2; leak*=2
            if name=='reflecting': leak=0
            if name=='strong_robin': leak*=2
            A[i,j]=-d; A[i,i]=d+leak
        return sparse.csc_matrix(A)
    m._axis_operator=variant
    S,_=m._forward(s); e=np.zeros_like(obs); slice_rmse=[]
    for coord in np.unique(ij,axis=0):
        test=np.all(groups==coord,axis=1); train=~test
        z=Z[train]; B=np.c_[S[train],obs[train]]
        q=np.linalg.solve(z.T@z+nr*np.eye(z.shape[1]),z.T@B)
        proj=B-z@q; A,y0=proj[:,:-1],proj[:,-1]
        w,_=m._box_simplex(A.T@A,-A.T@y0,cap,budget)
        e[test]=(S[test]-Z[test]@q[:,:-1])@w-(obs[test]-Z[test]@q[:,-1]); slice_rmse.append(np.sqrt(np.mean(e[test]**2)))
    print('boundary',name,'LOSO RMSE',np.sqrt(np.mean(e*e)),'worst sensor',max(slice_rmse),'worst shift',np.sqrt(np.mean(e.reshape(len(s['mu_shifts']),-1)**2,axis=1)).max())
