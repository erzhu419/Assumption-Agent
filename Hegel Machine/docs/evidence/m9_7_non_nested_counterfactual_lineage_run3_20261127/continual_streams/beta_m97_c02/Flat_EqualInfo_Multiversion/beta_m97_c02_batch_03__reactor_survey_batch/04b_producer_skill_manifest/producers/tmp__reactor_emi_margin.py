import importlib.util, itertools, pickle
from pathlib import Path
import numpy as np
p=Path('/holdout_task'); path=p/'beta_m97_c02_batch_03_solver.py'
spec=importlib.util.spec_from_file_location('candidate',path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load((p/'public_beta_m97_c02_batch_03.pkl').open('rb'))[0]['study']; sw={k:np.asarray(v) for k,v in s['sensor_calibration_sweeps'].items()}; baseline=m._calibrate(s)
for n, spike_count in ((12,1),(6,2)):
    worst=(0.,); failures=total=0
    for bank in range(4):
        bank_idx=np.flatnonzero(sw['bank_id']==bank); panels=[bank_idx] if n==12 else [bank_idx[:6],bank_idx[6:],bank_idx[::2],bank_idx[1::2]]
        for panel in panels:
            keep=(sw['bank_id']!=bank)|np.isin(np.arange(len(sw['bank_id'])),panel)
            for spike_rows in itertools.combinations(panel,spike_count):
                for signs in itertools.product((-1,1),repeat=spike_count):
                    raw=sw['raw_readout'].copy(); raw[list(spike_rows)]+=3.01*np.array(signs)
                    sweep={k:(raw if k=='raw_readout' else v)[keep] for k,v in sw.items()}
                    error=float(np.max(abs(m._calibrate(dict(s,sensor_calibration_sweeps=sweep))-baseline)))
                    worst=max(worst,(error,bank,panel.tolist(),spike_rows,signs)); failures+=error>1e-5; total+=1
    print(f'{n} rows, {spike_count} spikes: {failures}/{total} above 1e-5 calibrated-readout error; worst {worst}')
