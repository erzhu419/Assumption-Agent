from itertools import combinations
import importlib.util,pickle,numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',p+'alpha_m97_c01_batch_03_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open(p+'public_alpha_m97_c01_batch_03.pkl','rb'))[0]['dataset']; cal=d['calibration_load_cells']; base=m._calibrate(cal)
for size in (3,4):
    worst=0.; clean_error=0.; total=0
    for c in range(4):
        idx=np.flatnonzero(cal['lab_id']==c); others=np.flatnonzero(cal['lab_id']!=c)
        for picks in combinations(idx,size):
            subset=np.r_[list(picks),others]; q={k:v[subset].copy() for k,v in cal.items()}
            expected=np.polyfit(q['cmd_stress'][:size],q['true_stress'][:size],1)
            clean_error=max(clean_error,np.max(abs(m._calibrate(q)[c]-expected)))
            for i in range(size):
                old=q['true_stress'][i]
                for shift in (-4.,-3.01,3.01,4.):
                    q['true_stress'][i]=old+shift
                    worst=max(worst,np.max(abs(m._calibrate(q)[c]-base[c]))); total+=1
                q['true_stress'][i]=old
    print('sweep size',size,'EMI cases',total,'worst coefficient error',worst,'clean deviation from current fit',clean_error)
    assert clean_error < 1e-9 and worst < .001
