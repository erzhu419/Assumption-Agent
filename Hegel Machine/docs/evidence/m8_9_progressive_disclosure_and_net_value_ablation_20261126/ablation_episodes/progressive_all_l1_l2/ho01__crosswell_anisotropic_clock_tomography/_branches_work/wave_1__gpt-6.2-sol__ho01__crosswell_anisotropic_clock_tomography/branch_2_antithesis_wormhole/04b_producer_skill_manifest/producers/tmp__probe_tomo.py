import sys,pickle,time
sys.path.insert(0,'/holdout_task')
import numpy as np
from tomography_solver import _RayModel, _bound_clocks
for row in pickle.load(open('/holdout_task/public_surveys.pkl','rb')):
    a=_RayModel(row['survey']); t=time.monotonic(); p=a.solve(); dt=time.monotonic()-t
    e=a.y-a.design@p[3:]-p[2]*a.kernel(*p[:2]); c=_bound_clocks(a.clock@p[5:])/1000
    b=row['gt']; loc=np.linalg.norm(p[:2]-b['plume_xz_m']); clk=np.sqrt(np.mean((c-b['rcv_clock_bias_s'])**2))
    print('params x,z,R*delta_s,W*s0,W*s0*eps [m,ms]',p[:5], 'error [m,s]',loc,clk,'duration',dt)
    print('core MAD scale (ms)',1.4826*np.median(np.abs(e-np.median(e))),'skips',np.round(e[np.abs(e)>2.5],4))
    core=np.abs(e)<2.5
    for label, mask in [('upper',a.zs<np.median(a.zs)),('lower',a.zs>=np.median(a.zs)),('steep',np.abs(a.v)>.6),('shallow',np.abs(a.v)<=.6)]:
        k=mask&core; print(label,int(k.sum()),'rms ms',np.sqrt(np.mean(e[k]**2)))
