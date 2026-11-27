import importlib.util, itertools, pickle, numpy as np
r='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',r+'alpha_m97_c03_batch_02_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
p=pickle.load(open(r+'public_alpha_m97_c03_batch_02.pkl','rb'))[0]; d,g=p['dataset'],p['gt']; c=d['calibration_load_cells']; labs=c['lab_id']; base=np.array(m._calibrate(c)); original=m._ARCHIVE; o=m.solve_alpha_m97_c03_batch_02(d)
for out, truth in [('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]:
    e=(o[out]-g[truth])**2; missing=~np.isfinite(d['test']['coupon_features_raw']).all(axis=1); print(out,'RMSE complete/missing',np.sqrt(e[~missing].mean()),np.sqrt(e[missing].mean()))
for n in (3,4):
    worst={amp:(0.,None) for amp in (3.01,3.1,4.,8.)}; alarms=0
    for lab in range(4):
        for rows in itertools.combinations(np.flatnonzero(labs==lab),n):
            idx=list(rows); sub={k:v[idx].copy() for k,v in c.items()}; x=sub['cmd_stress']; y=sub['true_stress']; line=np.polyfit(x,y,1)
            m._ARCHIVE='/tmp/nonexistent_clean_calibration_history.json'
            try: actual=np.array(m._calibrate({k:np.concatenate((v[labs!=lab],sub[k])) for k,v in c.items()}))[:,lab]
            except FileNotFoundError: alarms+=1
            else: np.testing.assert_allclose(actual,line,atol=1e-9,rtol=0)
            finally: m._ARCHIVE=original
            if np.ptp(x)<0.5: continue
            sweep={k:np.concatenate((sub[k],v[labs!=lab])) for k,v in c.items()}
            for j in range(n):
                for amp in worst:
                    for spike in (-amp,amp):
                        sweep['true_stress'][j]=y[j]+spike
                        est=np.array(m._calibrate(sweep))[:,lab]; err=float(max(abs(est-base[:,lab])))
                        if err>worst[amp][0]: worst[amp]=(err,(lab,x.tolist(),j,spike,est.tolist()))
                sweep['true_stress'][j]=y[j]
    print(n,'rows: clean archive alarms',alarms,'worst errors by spike amplitude',worst)
