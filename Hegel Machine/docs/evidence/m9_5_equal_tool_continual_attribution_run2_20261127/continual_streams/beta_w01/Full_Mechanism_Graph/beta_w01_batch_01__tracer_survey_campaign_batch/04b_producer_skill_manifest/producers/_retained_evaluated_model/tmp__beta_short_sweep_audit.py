import importlib.util, pickle
import numpy as np
sp = importlib.util.spec_from_file_location('solver', '/holdout_task/beta_w01_batch_01_solver.py')
m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
for item in pickle.load(open('/holdout_task/public_beta_w01_batch_01.pkl','rb')):
    s, gt = item['study'], item['gt']; d,t = m._responses(s); z=s['Z_nuisance']
    sw=s['sensor_calibration_sweeps']; bank=sw['bank_id']; x=sw['ref_signal']; rows=s['row_bank_id']; y=m._calibrated_observations(s,d,z)
    S,r=gt['S_tilde'],gt['r_y']; cap=s['u_box']; ridge=s['source_ridge']; h=S.T@S+ridge*np.eye(S.shape[1])
    def audit(trial):
        rc=m._profile(z,m._calibrated_observations(trial,d,z)[:,None],s['nuisance_ridge']).ravel()
        result=[]
        for i,b in enumerate(s['budget_grid']):
            w,nu=m._box_simplex(h,-S.T@rc,cap,b); rr=S@w-r; g=S.T@rr+ridge*w
            result.append([g@w-nu*b-cap*np.minimum(g-nu,0).sum(), np.max(np.abs(S.T@(rc-r))), abs(.5*(rr@rr-(S@w-rc)@(S@w-rc))), np.max(np.abs(w-gt['weights_by_budget'][i])), np.max(np.abs((t@w).reshape(s['nx'],s['ny'])-gt['terminal_field_mid_budget'])) if i==1 else 0.])
        return np.max(result,axis=0)
    print('metrics: KKT gap, stationarity error, objective error, weight Linf, middle field Linf')
    print('public',audit(s))
    for label,gains,offsets in [('unchanged',np.ones(4),np.zeros(4)),('shifted',np.array([1.06,.96,1.18,.85]),np.array([.02,-.03,.06,-.04]))]:
        ids=[np.flatnonzero(bank==b) for b in range(4)]
        ids=[i[np.argsort(x[i])] for i in ids]
        for k in (1,3,4):
            vals=[]
            for wing in (0,1):
                group=ids[2*wing:2*wing+2]
                layouts=([[ind[j%len(ind)] for ind in group] for j in range(max(map(len,group)))] if k==1 else [[ind[np.linspace(0,len(ind)-1,k,dtype=int)] for ind in group]])
                for picks in layouts:
                    pick=np.array(picks).ravel(); take=(bank//2!=wing); take[pick]=True
                    for spike in [-1,*pick] if k>1 else [-1]:
                        yy=gains[bank]*x+offsets[bank];
                        if spike>=0: yy[spike]+=.75
                        trial=dict(s,y_obs_raw=gains[rows]*y+offsets[rows],sensor_calibration_sweeps=dict(bank_id=bank[take],ref_signal=x[take],raw_readout=yy[take])); vals.append(audit(trial))
            print(label,k,'points per sparse bank',np.max(vals,axis=0))
