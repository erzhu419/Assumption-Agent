import copy, importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m97_c03_batch_01_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_01.pkl','rb'))[0]['dataset']; s=d['train']; n=len(s['lab_id']); t,dt=d['eval_times'],d['past_dt']; p,h,f,b=m._prepare(d); damage=np.maximum(0,abs(p)-d['crit_stress'])**2@dt; y=s['strain_curves_obs']-b[:n,None]; r=s['instant_rate_obs']; complete=np.isfinite(s['coupon_features_raw']).all(1); lab=s['lab_id']
# Fit challenger coefficients exclusively on each fold's training labs. They change only the log-clock amplitude and its matching initial derivative.
for leave in range(4):
    tr=lab!=leave; va=(lab==leave)&complete; idx=np.flatnonzero(va)
    tau,w=m._fit(p[:n][tr],h[:n][tr],f[:n][tr],dt,t,damage[:n][tr],y[tr],r[tr],True)
    A,B=m._design(tau,p[:n],h[:n],f[:n],dt,t,damage[:n],True); ec=y-A@w; er=r-B@w
    z=f[:n,1:]; diff=p[:n,-1]-h[:n]; pair=np.column_stack([z[:,i]*z[:,j] for i in range(4) for j in range(i,4)])
    rational=np.column_stack([h[:n]/(1+diff**2),diff/(1+h[:n]**2),np.tanh(damage[:n])*diff/(1+diff**2)])
    options={'specified':(ec,er)}
    for name,F in [('pairwise',pair),('rational',rational)]:
        C=F[:,None,:]*np.log1p(t)[None,:,None]; iy=tr[:,None]&complete[:,None]&np.isfinite(ec); ir=tr&complete&np.isfinite(er)
        scales=[max(1e-6,1.4826*np.median(abs(v-np.median(v)))) for v in [ec[iy],er[ir]]]
        M=np.vstack([C[iy]/scales[0],F[ir]/scales[1]]); q=np.r_[ec[iy]/scales[0],er[ir]/scales[1]]
        coeff=least_squares(lambda x:M@x-q,np.zeros(F.shape[1]),loss='cauchy',f_scale=2).x
        options[name]=(ec-C@coeff,er-F@coeff)
    for name,(c,e) in options.items():
        print('held lab',leave,name,'strain/rate p85',np.nanquantile(abs(c[idx]),.85),np.nanquantile(abs(e[idx]),.85),'signed rate medians by stratum',[round(float(np.nanmedian(e[idx][s['stratum'][idx]==g])),5) for g in np.unique(s['stratum'][idx])])
# Unit and origin changes per lab must leave aligned inputs and outputs unchanged.
d2=copy.deepcopy(d); gain=np.array([1.4,.65,2.,.8]); shift=np.array([-.9,1.3,.5,-2.]); cg=np.array([.8,1.3,.7,1.1]); cb=np.array([.6,-.9,.3,.2]); c=d2['calibration_load_cells']; j=c['lab_id']; c['cmd_stress']=cg[j]*c['cmd_stress']+cb[j]
for key in ('train','test'):
    a=d2[key]; j=a['lab_id']; a['ref_coupons']=gain[j,None]*a['ref_coupons']+shift[j,None]; a['coupon_features_raw']=gain[j,None]*a['coupon_features_raw']+shift[j,None]; a['past_stress_cmd']=cg[j,None]*a['past_stress_cmd']+cb[j,None]; a['stress_hold_cmd']=cg[j]*a['stress_hold_cmd']+cb[j]
for a,c in zip(m._prepare(d),m._prepare(d2)): np.testing.assert_allclose(a,c,atol=1e-11,rtol=0)
a,c=m.solve_alpha_m97_c03_batch_01(d),m.solve_alpha_m97_c03_batch_01(d2)
for k in a:
    print('lab affine invariance',k,float(np.max(abs(a[k]-c[k])))); np.testing.assert_allclose(a[k],c[k],atol=1e-5,rtol=0)
