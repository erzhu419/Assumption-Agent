import importlib.util
import pathlib
import pickle
import numpy as np

st = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
archive = pathlib.Path('/tmp/hegel_continual_archive')
cal = st['sensor_calibration_sweeps']
x = np.column_stack((np.ones(len(cal['ref_rosette'])), cal['ref_rosette']))
yraw = cal['raw_rosette']
e = []
for i in range(len(x)):
    tr = np.arange(len(x)) != i
    a = np.linalg.lstsq(x[tr], yraw[tr], rcond=None)[0]
    e.append(x[i] @ a - yraw[i])
a = np.linalg.lstsq(x, yraw, rcond=None)[0]
print('affine calibration LOO RMSE by channel', np.mean(np.square(e), axis=0)**.5, 'condition', np.linalg.cond(a[1:]))
ns = len(st['sensor_ij'])
print('sensor count/unique', ns, len(np.unique(st['sensor_ij'],axis=0)))
print('wall sensor count', np.sum(np.any((st['sensor_ij']==0)|(st['sensor_ij']==[st['nx']-1,st['ny']-1]),axis=1)))
for p in sorted(archive.glob('*solver.py')):
    spec = importlib.util.spec_from_file_location(p.stem,p)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    d,_ = m._forward(st); y=m._calibrate(st); z=st['Z_nuisance']
    zr = st['nuisance_ridge']*np.eye(z.shape[1]); wr=st['source_ridge']*np.eye(d.shape[1]); n=len(y)
    def fit(tr, te):
        D,Y,Z=d[tr],y[tr],z[tr]; q=np.column_stack((D,Y)); gram=Z.T@Z+zr
        q-=Z@np.linalg.solve(gram,Z.T@q); S,R=q[:,:-1],q[:,-1]
        w,_=m._box_simplex_qp(S.T@S+wr,-S.T@R,st['u_box'],st['total_budget']); f=D@w-Y
        gamma=-np.linalg.solve(gram,Z.T@f)
        return d[te]@w+z[te]@gamma-y[te]
    e=np.empty(n)
    for j in range(ns):
        te=np.arange(n)%ns==j; e[te]=fit(~te,te)
    ec=np.empty(n)
    for j in range(0,n,4):
        te=(np.arange(n)>=j)&(np.arange(n)<j+4); ec[te]=fit(~te,te)
    out=m.solve_batch(st); k=out['kkt_certificate']; print(p.stem, 'obj',k['primal_objective'],'gap',k['primal_objective']-k['dual_lower_bound'],'sensor CV RMSE',np.mean(e**2)**.5,'max sensor RMSE',np.max(np.mean(e.reshape(-1,ns)**2,axis=0)**.5),'shift RMSE',np.mean(e.reshape(-1,ns)**2,axis=1)**.5,'rosette CV RMSE',np.mean(ec**2)**.5)
