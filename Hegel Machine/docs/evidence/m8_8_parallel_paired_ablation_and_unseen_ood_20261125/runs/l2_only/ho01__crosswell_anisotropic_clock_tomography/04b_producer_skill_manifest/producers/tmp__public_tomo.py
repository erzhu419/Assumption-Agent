import pickle,time
import numpy as np
from tomography_solver import _fit,invert_plume_and_clocks
samples=pickle.load(open('/holdout_task/public_surveys.pkl','rb'))
for i,sample in enumerate(samples):
    s,gt=sample['survey'],sample['gt']; t=time.monotonic(); r=invert_plume_and_clocks(s); elapsed=time.monotonic()-t
    rays,p,e=_fit(s); good=abs(e)<2; skip=e[~good]; print(i,'params',p[:5], 'actual scipy time',elapsed,'errors',np.linalg.norm(np.subtract(r['plume_xz_m'],gt['plume_xz_m'])), np.sqrt(np.mean(np.subtract(r['rcv_clock_bias_s'],gt['rcv_clock_bias_s'])**2)),'clean rms',np.sqrt(np.mean(e[good]**2)),'skips',np.round(skip,4),'clock sum',sum(r['rcv_clock_bias_s']))
