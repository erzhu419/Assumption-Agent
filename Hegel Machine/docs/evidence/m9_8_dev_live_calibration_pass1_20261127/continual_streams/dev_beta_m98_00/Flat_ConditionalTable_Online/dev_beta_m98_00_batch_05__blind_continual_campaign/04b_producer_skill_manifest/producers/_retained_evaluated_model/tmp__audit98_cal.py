import importlib.util, pickle
import numpy as np
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=np.asarray(c['ref_rosette']); raw=np.asarray(c['raw_rosette']); rng=np.random.default_rng(98); folds=np.array_split(rng.permutation(len(x)),5); folds.append(np.argsort(np.linalg.norm(x,axis=1))[-16:]); bank=c['single_bank_sweeps']; ids=np.asarray(bank['bank_id']); bx=np.eye(4)[ids]*np.asarray(bank['ref_signal'])[:,None]; br=np.asarray(bank['raw_readout']); records={k:[] for k in ('affine','pair_polynomial','contrast_rational')}
def basis(x, quadratic=False):
    terms=[np.ones(len(x)),*x.T]
    if quadratic: terms += [x[:,i]*x[:,j] for i in range(4) for j in range(i,4)] + [x[:,i]**3 for i in range(4)]
    return np.column_stack(terms)
for test in folds:
    tr=np.ones(len(x),bool); tr[test]=False
    for j in range(4):
        own=ids==j; xx=np.r_[x[tr],bx[own]]; yy=np.r_[raw[tr,j],br[own]]; xt=x[test]; yt=raw[test,j]
        for name in ('affine','pair_polynomial'):
            quad=name!='affine'; coef=np.linalg.lstsq(basis(xx,quad),yy,rcond=None)[0]; records[name].append(np.sqrt(np.mean((basis(xt,quad)@coef-yt)**2)))
        # Choose the contrast and denominator strength using training rows only.
        inner=np.arange(len(xx))%4; best=(np.inf,None)
        for k in range(4):
            if k==j: continue
            for sign in (-1,1):
                for a in (0.,1e-5,1e-4,1e-3,1e-2,.1):
                    def design(v): return basis(v,True)/(1+a*(v[:,j]+sign*v[:,k])**2)[:,None]
                    D=design(xx); errors=[]
                    for f in range(4):
                        keep=inner!=f; coef=np.linalg.lstsq(D[keep],yy[keep],rcond=None)[0]; errors.extend((D[~keep]@coef-yy[~keep])**2)
                    loss=np.mean(errors)
                    if loss<best[0]: best=(loss,(k,sign,a))
        k,sign,a=best[1]; D=basis(xx,True)/(1+a*(xx[:,j]+sign*xx[:,k])**2)[:,None]; T=basis(xt,True)/(1+a*(xt[:,j]+sign*xt[:,k])**2)[:,None]; coef=np.linalg.lstsq(D,yy,rcond=None)[0]; records['contrast_rational'].append(np.sqrt(np.mean((T@coef-yt)**2)))
for key, vals in records.items(): print(key,'worst held sweep/channel RMSE',max(vals),'outer RMS',np.sqrt(np.mean(np.square(vals))),'high-amplitude worst',max(vals[-4:]))
cal=m._calibration(c); y=cal(s['y_obs_raw']); S,_=m._response(s,np.ones(8),s['source_modes']); Z=s['Z_nuisance']; PZ=np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T); X=S-Z@(PZ@S); t=y-Z@(PZ@y); w=m.solve_batch(s)['weights']; rr=(X@w-t).reshape(len(s['mu_shifts']),-1,4);print('nuisance-profiled RMSE by shift and channel:',np.sqrt(np.mean(rr**2,axis=1))); print('target calibration range inside sweeps',np.all(y.reshape(-1,4)>=x.min(0)) and np.all(y.reshape(-1,4)<=x.max(0))); print('weight sensitivity bound per unit calibrated error',np.linalg.norm(X.T@(np.eye(len(y))-Z@PZ),2)/s['source_ridge'])
