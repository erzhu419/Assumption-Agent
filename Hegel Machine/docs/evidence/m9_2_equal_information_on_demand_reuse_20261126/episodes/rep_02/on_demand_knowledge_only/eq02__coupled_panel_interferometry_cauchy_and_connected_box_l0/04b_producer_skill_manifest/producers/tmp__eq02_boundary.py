import runpy, time
import numpy as np
q=runpy.run_path('/tmp/eq02_check.py'); replay=q['replay']; mod=q['mod']; make=q['make']
worst=np.zeros(4); times=[]
for seed in range(8):
    p=18+seed%2
    st,z,theta=make(80+seed,p,.94 if seed<4 else .985,True)
    X=st['X_cand']; st['lam0']=(.018,.035,.065,.105)[seed%4]; st['lam2']=.025 if seed<4 else .006; st['M']=.45 if seed%2 else .7
    # Increase the target-only signal on two similar columns to exercise box saturation and ambiguous supports.
    st['y_assay_raw'][st['stage3_indices']]+=1.8*X[:,1]+1.3*X[:,2]-1.1*X[:,6]
    t=time.perf_counter(); out=mod.solve_coupled_interferometry_l0(st); t=time.perf_counter()-t; times.append(t)
    c=out['certified_selection']; leaves,gap,_=replay(X,out['deconfounded_target'],st['lam0'],st['lam2'],st['M'],c)
    mae=np.mean(abs(out['aligned_panel']-z)); rel=np.linalg.norm(out['calibration_weights']-theta)/np.linalg.norm(theta)
    saturated=np.count_nonzero(np.isclose(abs(np.asarray(c['weights'])),st['M']))
    print('CONNECTED BOX',seed,p,'leaves',leaves,'gap',gap,'saturated',saturated,'seconds',round(t,3),flush=True)
    worst=np.maximum(worst,[mae,rel,gap,leaves])
assert worst[0]<=.11 and worst[1]<=.14 and sum(sorted(times)[-2:])<=18
print('WORST mae, calibration rel-L2, certificate gap, leaves:',worst.tolist(),'slowest pair seconds:',sum(sorted(times)[-2:]))
