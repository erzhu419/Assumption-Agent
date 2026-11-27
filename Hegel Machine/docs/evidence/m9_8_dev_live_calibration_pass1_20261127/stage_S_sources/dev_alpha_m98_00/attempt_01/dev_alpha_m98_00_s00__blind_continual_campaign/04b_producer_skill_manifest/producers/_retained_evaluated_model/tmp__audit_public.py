import importlib.util, pickle, pathlib, numpy as np
s=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; tr,te,pi=(p[k] for k in ('train','test','chamber_verification_trials')); dt,t=p['past_dt'],p['eval_times']
archive=pathlib.Path('/tmp/hegel_continual_archive'); print('archive files',[(f.name,f.stat().st_size) for f in archive.iterdir() if f.is_file()] if archive.is_dir() else None)
log=archive/'archive_apparatus_log.json'
if log.is_file():
    with log.open() as f: print('apparatus log preview',f.read(3500))
c=p['calibration_load_cells']; raw,true=c['bridge_raw'],c['true_channel_loads']; single=np.count_nonzero(true,axis=1)==1
# Coherent nonlinear bridge terms must improve independently held-out sweep geometries.
def bridge(x, mode):
    cols=[np.ones(len(x)),*x.T]
    if mode=='quadratic': cols += [x[:,i]*x[:,j] for i in range(4) for j in range(i,4)]
    if mode=='rational':
        cols += [(x[:,i]+sign*x[:,j])/(1+.25*(x[:,i]-sign*x[:,j])**2) for i in range(4) for j in range(i+1,4) for sign in (-1,1)]
    return np.column_stack(cols)
fold=np.arange(len(raw))%5
for mode in ('affine','quadratic','rational'):
    a=bridge(raw,mode); e=np.zeros(len(raw))
    for k in np.unique(fold):
        keep=fold!=k; b=np.linalg.lstsq(a[keep],true[keep].sum(axis=1),rcond=None)[0]; e[~keep]=a[~keep]@b-true[~keep].sum(axis=1)
    print('bridge',mode,'single/mixed p95',*[round(np.quantile(abs(e[mask]),.95),7) for mask in (single,~single)])
cal=m._calibration(c); a,b=[m._inputs(z,x,cal) for z,x in zip((tr,te),m._features(tr,te))]; d=m._inputs(pi,m._features(pi,balanced=False)[0],cal)
tau,th,scale=m._material(tr,a,dt,t)
# Check order and initial derivative independently against a literal recurrence.
z=np.zeros((len(b[0]),2))
for delta,stress in zip(dt,b[0].T):
    decay=np.exp(-delta/tau); z=decay*z+(1-decay)*stress[:,None]
_,r=m._predict(b,dt,t,tau,th); print('recurrence state and rate discrepancy',np.max(abs(z-b[0]@m._kernel(dt,tau))),np.max(abs(r-((b[1][:,None]-z)*th[:2]/tau).sum(axis=1)-b[2]@th[2:])))
# Check supported real slices: late-time drift and rates, by lab and missingness.
for name,blk,inp in [('train',tr,a),('pilot',pi,d)]:
    er=np.empty(len(blk['lab_id'])); ec=np.empty_like(blk['strain_curves_obs'])
    for lab in np.unique(blk['lab_id']):
        mask=blk['lab_id']==lab; ta,tt=(tau,th) if name=='pilot' else m._material(tr,a,dt,t,~mask)[:2]; yy,rr=m._predict(tuple(v[mask] for v in inp),dt,t,ta,tt); ec[mask]=blk['strain_curves_obs'][mask]-blk['strain_0'][mask,None]-yy; er[mask]=blk['instant_rate_obs'][mask]-rr
        print(name,'lab',lab,'early/late p90',np.round(np.quantile(abs(ec[mask][:,[0,-1]]),.9,axis=0),6),'rate rmse',round(np.sqrt(np.mean(er[mask]**2)),6))
    missing=np.isnan(blk['coupon_features_raw']).any(axis=1)
    for mask in (missing,~missing):
        if mask.any(): print(name,'slice size',int(mask.sum()),'late signed median',round(np.median(ec[mask,-1]),6),'rate rmse',round(np.sqrt(np.mean(er[mask]**2)),6))
