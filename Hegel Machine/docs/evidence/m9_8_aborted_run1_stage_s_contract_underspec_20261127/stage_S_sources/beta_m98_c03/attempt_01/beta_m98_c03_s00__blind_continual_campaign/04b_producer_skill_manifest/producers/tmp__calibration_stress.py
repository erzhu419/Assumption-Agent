import importlib.util,pickle,numpy as np
sp=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['sensor_calibration_sweeps']; x=d['ref_rosette']; b=d['single_bank_sweeps']; j=b['bank_id']; z=np.eye(4)[j]*b['ref_signal'][:,None]; rng=np.random.default_rng(427); test=rng.uniform(-.65,.85,(256,4)); M=np.array([[1.13,.13,0,-.05],[-.04,.9,.05,0],[0,.09,1.05,-.05],[.03,0,-.1,1.2]]); off=np.array([.06,-.12,.03,.09]);
def basic(t): return t@M+off
pairs=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]; C=rng.uniform(-.06,.06,(6,4));
for name,f in [('affine',basic),('separable',lambda t:basic(t)+.09*t*t-.04*t*t*t),('coupled',lambda t:basic(t)+.09*t*t+np.column_stack([t[:,a]*t[:,b] for a,b in pairs])@C)]:
 sweep=dict(ref_rosette=x,raw_rosette=f(x)+rng.normal(0,1e-5,x.shape),single_bank_sweeps=dict(b,raw_readout=f(z)[np.arange(len(j)),j])); c=m._calibrator(sweep); r=c(f(test)).reshape(-1,4)-test; print(name,'worst channel inverse RMSE',np.sqrt((r*r).mean(axis=0)).max(),'max abs',np.abs(r).max());
