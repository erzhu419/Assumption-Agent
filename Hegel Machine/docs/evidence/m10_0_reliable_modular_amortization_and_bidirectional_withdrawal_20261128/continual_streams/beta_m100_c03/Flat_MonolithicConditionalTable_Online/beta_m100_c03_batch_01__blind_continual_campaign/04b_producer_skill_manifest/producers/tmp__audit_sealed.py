import copy, importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares

root='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',root+'batch_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
items=pickle.load(open(root+'public_batch.pkl','rb'))
for j,item in enumerate(items):
    s=item['study']; c=s['sensor_calibration_sweeps']; x=np.asarray(c['ref_rosette']); y=np.asarray(c['raw_rosette']); x=(x-x.mean(0))/x.std(0)
    X=np.column_stack((np.ones(len(x)),x)); pairs=[(a,b) for a in range(4) for b in range(a+1,4)]
    XX=np.column_stack((X,x*x,*[x[:,a]*x[:,b] for a,b in pairs]))
    errors=[]
    for test in np.array_split(np.arange(len(x)),5):
        train=np.setdiff1d(np.arange(len(x)),test)
        B=np.linalg.lstsq(X[train],y[train],rcond=None)[0]
        D=np.linalg.lstsq(XX[train],y[train],rcond=None)[0]
        # Per-channel saturating gain: affine numerator, fitted squared-input denominator.
        def rat(p,idx):
            return (X[idx]@p[:-4].reshape(5,4))/(1+x[idx]**2*p[-4:])
        fit=least_squares(lambda p:(rat(p,train)-y[train]).ravel(),np.r_[B.ravel(),np.zeros(4)],bounds=(np.r_[np.full(20,-np.inf),np.zeros(4)],np.inf),gtol=1e-11)
        errors.append([np.sqrt(np.mean((pred-y[test])**2,axis=0)).max() for pred in (X[test]@B,XX[test]@D,rat(fit.x,test))])
    print('item',j,'sweep worst blocked RMSE affine/pair-interaction/rational',np.max(errors,axis=0))
    S,U=m._responses(s)
    pulses=item.get('campaign_diagnostics') or []
    if pulses:
        errors=np.stack([m._calibrate(s,p['y_pulse_raw'])-S@p['w_pulse'] for p in pulses]); print('worst pulse by (shift,rosette) RMSE',np.sqrt(np.mean(errors.reshape(len(pulses),len(s['mu_shifts']),-1,4)**2,axis=-1)).max())
        if 'row_bank_id' in s:
            bank=np.asarray(s['row_bank_id']).ravel()
            print('pulse RMSE within banks',[(str(b),float(np.sqrt(np.mean(errors[:,bank==b]**2)))) for b in np.unique(bank)])
    # Independent application of the sealed transport stencil to the returned field.
    def apply_axis(f,d):
        a=f if d=='x' else f.T
        k=np.asarray(s['kappa_'+d]); v=np.asarray(s['vel_'+d]); h=s['L'+d]/(len(k)-1)
        dd=(a[1:]-a[:-1])/h
        flux=.5*(k[1:]+k[:-1])[:,None]*dd
        z=np.empty_like(a); z[0]=-k[0]*dd[0]/h; z[-1]=k[-1]*dd[-1]/h
        z[1:-1]=(flux[:-1]-flux[1:])/h+np.maximum(v[1:-1],0)[:,None]*dd[:-1]+np.minimum(v[1:-1],0)[:,None]*dd[1:]
        return z if d=='x' else z.T
    cap=s['u_box']; total=len(s['source_modes'])*cap; worst=np.zeros(4)
    for budget in (0.,.03*total,.5*total,.97*total,total,s['total_budget']):
        st=copy.copy(s); st['total_budget']=budget
        o=m.solve_batch(st); w=o['weights']; f=o['terminal_field']; k=o['kkt_certificate']; lo=np.asarray(k['lam_lower']); hi=np.asarray(k['lam_upper']); nu=k['nu_sum']
        rhs=np.tensordot(w,s['source_modes'],axes=(0,0))
        fp=apply_axis(f,'x')+apply_axis(f,'y')+(s['V_reaction_2d']+s['mu_shifts'][-1])*f
        z=np.asarray(s['Z_nuisance']); D=np.column_stack((S,m._calibrate(s,s['y_obs_raw']))); D-=z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@D); A,ycal=D[:,:-1],D[:,-1]; r=A@w-ycal
        g=A.T@r+s['source_ridge']*w; J=.5*(r@r+s['source_ridge']*(w@w)); dual=J-g@w+nu*budget+cap*np.minimum(0,g-nu).sum()
        assert min(w)>=0 and max(w)<=cap and min(lo)>=0 and min(hi)>=0 and np.array_equal(w,k['weights'])
        assert abs(k['dual_lower_bound']-dual)<1e-10 and abs(k['primal_objective']-J)<1e-10
        worst=np.maximum(worst,[abs(w.sum()-budget),np.max(np.abs(fp-rhs)),np.max(np.abs(g-nu-lo+hi)),max(abs(J-dual),np.max(abs(lo*w)),np.max(abs(hi*(cap-w))))])
    print('budget/field PDE/stationarity/complementarity-gap worst',worst)
    assert np.all(worst<1e-9)
