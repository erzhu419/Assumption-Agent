import importlib.util,itertools,pickle
import numpy as np
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/beta_w04_batch_01_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_beta_w04_batch_01.pkl','rb'))[0]['study']
bank,ref,raw=m._sweep_arrays(s['sensor_calibration_sweeps']); history=m._historical_calibration(list(range(4))); pairs=np.triu_indices(3,1)
def options(x,y,prior):
    a,b=pairs; delta=x[b]-x[a]; keep=np.abs(delta)>1e-12*max(1.,np.max(abs(x))); a,b,delta=a[keep],b[keep],delta[keep]
    gains=(y[b]-y[a])/delta; keep=gains>0; gains,a,b=gains[keep],a[keep],b[keep]
    offsets=y[a]-gains*x[a]; g0,c0=prior
    base=m._fit_bank(x,y); g,c=base
    if np.max(np.abs(y-g*x-c)) <= 1e-6*max(1.,np.max(np.abs(y))) or not gains.size:
        return [base]*3
    dist=np.abs(np.log(gains/g0)); joint=np.hypot(dist,(offsets-c0)/max(np.max(abs(x)),0.5)/g0)
    return [base, (gains[np.argmin(dist)],offsets[np.argmin(dist)]), (gains[np.argmin(joint)],offsets[np.argmin(joint)])]
for scale,offset in [(1.,0.),(1.18,.07),(.83,-.07),(1.18,-.07),(.83,.07)]:
    for amp in (0.,.8,-.8):
        worst=np.zeros(3); bad=np.zeros(3,dtype=int); total=0
        for b in range(4):
            ids=np.flatnonzero(bank==b); obs=s['y_obs_raw'][s['row_bank_id']==b]
            for take in itertools.combinations(ids,3):
                sel=list(take); x=ref[sel]; clean=scale*raw[sel]+offset
                for j in range(3) if amp else (0,):
                    y=clean.copy(); y[j]+=amp
                    errs=np.array([np.max(np.abs((scale*obs+offset-c)/g-obs)) for g,c in options(x,y,history[b])]); worst=np.maximum(worst,errs); bad+=errs>1e-4; total+=1
        print('gain/offset/spike',scale,offset,amp,'worst',worst,'failures over 1e-4',bad,'of',total)
