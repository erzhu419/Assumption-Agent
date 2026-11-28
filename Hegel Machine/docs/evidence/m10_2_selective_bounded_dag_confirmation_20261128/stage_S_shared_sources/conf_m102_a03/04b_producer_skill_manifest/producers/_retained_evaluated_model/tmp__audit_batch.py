import importlib.util, pickle, numpy as np
from scipy.optimize import least_squares
p=importlib.util.spec_from_file_location('bs','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(p); p.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,T=[s['train'][k] for k in ('x','y','traj')]; h=s['eval_times']; a=m._calibration(s['panel_a']); b=m._fit(m._affine(s['panel_b']['u']),s['panel_b']['v']); z=m._affine(np.linalg.solve(a[1:].T,(x-a[0]).T).T)@b
print('Available fields:',sorted(s), 'row-level training fields:',sorted(s['train'])); print('Telemetry shape:',np.shape(s['telemetry_vec'])); print('B Gram eigenvalues:',np.linalg.eigvalsh(b[1:].T@b[1:]))
P=z[:,0]*z[:,2]; Q=z[:,1]*z[:,3]; X0=m._readout(z); z1=m._affine(z); norm=np.linalg.norm(z,axis=1)
def exp_modes(train):
    q,_=np.linalg.qr(z[train]); C=q.T@T[train]
    def basis(k): return -np.expm1(-h[:,None]*np.asarray(k)[None,:])
    def residual(k):
        D=basis(k); return (D@m._fit(D,C.T)-C.T).ravel()
    opt=least_squares(residual,[.35,1.8],bounds=([.001,.7],[1.4,10.]),ftol=1e-11,gtol=1e-11,xtol=1e-11)
    D=basis(opt.x); W=m._fit(z[train],T[train]); return m._fit(D,W.T).T@D.T,opt.x
# Rational challenger: pick scale and quadratic denominator inside the outer fold only.
den=[zz**2 for zz in z.T]+[(P+Q)**2,(P-Q)**2,norm**2]; rational=[X0]+[X0/(1+g*d[:,None]) for d in den for g in (.001,.003,.01,.03,.10)]
folds=[('required',np.arange(52,72)),('high-norm',np.argsort(norm)[-20:]),('large-interaction',np.argsort(abs(P+Q))[-20:])]+[(f'block{i}',q) for i,q in enumerate(np.array_split(np.arange(72),4))]
maxes={}
for name,te in folds:
    tr=np.setdiff1d(np.arange(72),te); sub={**s,'train':{k:s['train'][k][tr] for k in ('x','y','traj')},'test':{'x':x[te]}}; out=m.solve_batch(sub)
    mae=lambda v:np.mean(np.abs(v-y[te])); mse=lambda A,tt,v:np.mean(np.abs(A[v]@m._fit(A[tt],y[tt])-y[v])); grouped=np.array_split(tr,4)
    ridx=np.argmin([sum(mse(A,np.setdiff1d(tr,v),v)*len(v) for v in grouped)/len(tr) for A in rational]); C,k=exp_modes(tr); E=np.abs(z[te]@C-T[te]); F=np.abs(out['state_pred']-T[te]); vals=dict(affine=mse(z1,tr,te),bilinear=mae(out['y_hat']),signed_difference=mse(np.c_[z,P-Q],tr,te),free_pair=mse(np.c_[z,P,Q],tr,te),rational=mse(rational[ridx],tr,te),rank2=F.mean(),rank2_peak=F.mean(0).max(),exponential=E.mean(),exponential_peak=E.mean(0).max()); print(name,{key:round(value,6) for key,value in vals.items()},'rational_idx',ridx,'rates',k)
    for key,value in vals.items(): maxes[key]=max(value,maxes.get(key,0))
    assert max(vals['bilinear'],vals['rank2_peak'])<.038
print('worst slices',maxes)
