import glob,importlib.util,pickle,json
import numpy as np

def load(p):
    s=importlib.util.spec_from_file_location('probe',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

root='/tmp/hegel_continual_archive/'
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; st=item['study']; ps=item['campaign_diagnostics']
sw=st['sensor_calibration_sweeps']; X=sw['ref_rosette']; Y=sw['raw_rosette']; bank=sw['single_bank_sweeps']; Z=st['Z_nuisance']
A=np.column_stack((X,np.ones(len(X)))); B=np.linalg.lstsq(A,Y,rcond=None)[0]
print('cal full coefficients',np.round(B,5),'residual rms by channel',np.sqrt(np.mean((A@B-Y)**2,axis=0)))
print('single bank affine fits:')
for j in np.unique(bank['bank_id']):
    mask=bank['bank_id']==j; x=bank['ref_signal'][mask]; y=bank['raw_readout'][mask]; a=np.c_[x,np.ones(len(x))]; b=np.linalg.lstsq(a,y,rcond=None)[0]; print(j,np.round(b,5),np.linalg.norm(a@b-y)/np.sqrt(len(y)))
print('row bank first 24',st['row_bank_id'][:24],'edge sensor count',sum((st['sensor_ij']==0).any(1)| (st['sensor_ij']==[st['nx']-1,st['ny']-1]).any(1)))
for q in json.load(open(root+'archive_apparatus_log.json')):
    a=np.asarray(q['facility_telemetry_trace']); print('archive mean, last',q['completed_batch_index'],a.mean(0).round(4),a[-1].round(4))
a=st['aux_manifold_telemetry'];print('current telemetry mean,last',a.mean(0).round(4),a[-1].round(4))
for p in sorted(glob.glob(root+'*_solver.py')):
    m=load(p); S,_=m._forward(st); print(p.rsplit('/',1)[-1]);
    cals={'full affine':lambda y:np.linalg.solve(B[:4].T,(y.reshape(-1,4)-B[4]).T).T.ravel()}
    for name in ('_calibrate','_calibrate_observations','_calibrate_y'):
        if hasattr(m,name):
            import inspect
            print(name,inspect.signature(getattr(m,name)))
    for name,cal in cals.items():
        for q in ps:
            e=cal(q['y_pulse_raw'])-S@q['w_pulse']; ep=e-Z@np.linalg.lstsq(Z,e,rcond=None)[0]; print(name,'pulse norm/rms/removing Z',np.linalg.norm(e),np.sqrt(np.mean(e**2)),np.linalg.norm(ep))
