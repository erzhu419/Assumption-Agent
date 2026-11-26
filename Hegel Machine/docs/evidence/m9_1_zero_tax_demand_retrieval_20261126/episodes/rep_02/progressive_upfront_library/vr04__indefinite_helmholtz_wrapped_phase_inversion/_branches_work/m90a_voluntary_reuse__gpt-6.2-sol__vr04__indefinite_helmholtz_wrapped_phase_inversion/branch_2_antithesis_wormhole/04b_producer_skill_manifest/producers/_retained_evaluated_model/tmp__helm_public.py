import runpy, pickle
import numpy as np
m=runpy.run_path('/holdout_task/helmholtz_phase_inversion.py')
with open('/holdout_task/public_helmholtz_cases.pkl','rb') as f:
    cases=pickle.load(f)
rel=lambda x,y: np.linalg.norm(x-y)/np.linalg.norm(y)
for case in cases:
    p,gt=case['problem'],case['gt']; s=p['survey']; u=m['_responses'](p); i,j=s['sensor_ij'].T; a=u[i,j]; y=m['_phasors'](s)*np.exp(-1j*gt['global_phase_rad']); x=gt['source_amplitudes']; obs=np.abs(y-a@x)
    est=m['invert_helmholtz_sources_and_phase'](p); ls=np.linalg.lstsq(np.r_[a.real,a.imag],np.r_[y.real,y.imag],rcond=None)[0]
    jack=np.array([m['_fit'](a[np.arange(len(y))!=l],y[np.arange(len(y))!=l],s['tikhonov_alpha']) for l in range(len(y))]); se=max(rel(t,x) for t in jack); fe=max(rel(np.abs(u@t),gt['predicted_field_abs']) for t in jack)
    print('truth forward',rel(np.abs(u@x),gt['predicted_field_abs']),'condition',np.linalg.cond(np.r_[a.real,a.imag]),'sv',np.linalg.svd(np.r_[a.real,a.imag],compute_uv=False),'true residual quantiles',np.quantile(obs,[0,.5,.8,.9,.93,.95,1])); print('OLS source',rel(ls,x),'robust source',rel(est['source_amplitudes'],x),'robust field',rel(est['predicted_field_abs'],gt['predicted_field_abs']),'phase',abs(np.angle(np.exp(1j*(est['global_phase_rad']-gt['global_phase_rad'])))),'jackknife source,field',se,fe)
