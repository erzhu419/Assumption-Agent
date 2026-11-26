exec(open('/tmp/alpha_w02_batch04_cv.py').read().split('for a in s._archive_checks')[0])
from itertools import combinations
labs = tr['lab_id']; dt=d['past_dt']; time=d['eval_times']; sy=d['yield_stress']
y=np.column_stack((tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']))
x=s._features(tr,d['test'])[:len(y)]; st=s._stress(tr,cal); p=s._plastic(st,dt,sy)
for lab in range(4):
    fit=labs!=lab; val=~fit
    tau,b=s._fit(tuple(a[fit] for a in st),x[fit],tuple(a[fit] for a in p),dt,time,y[fit])
    r=y-s._design(st,x,p,dt,time,tau)@b
    # Predeclared smooth alternatives: saturation of each plastic term, and two-way/three-way feature interactions in the creep drift.
    hold=st[1]; ov,hist=p
    q=np.column_stack((ov/(1+np.maximum(abs(hold)-sy,0)**2),hist/(1+(hist/(1+abs(hold)))**2)))
    z=x[:,1:]; w=np.column_stack([z[:,i]*z[:,j] for i,j in combinations(range(4),2)]+[z[:,i]*z[:,j]*z[:,k] for i,j,k in combinations(range(4),3)]+[(z[:,0]+z[:,1])**2,(z[:,2]-z[:,3])**2]); w-=w[fit].mean(0)
    ramp=np.r_[time,1.]; rise=np.r_[-np.expm1(-time/tau[0]),1/tau[0]]
    alt=np.concatenate((q[:,0,None,None]*ramp[None,:,None],q[:,1,None,None]*rise[None,:,None],w[:,None,:]*ramp[None,:,None]),axis=2)
    std=np.maximum(1.4826*np.median(abs(r[fit]),axis=0),1e-6)
    a=(alt[fit]/std[None,:,None]).reshape(-1,alt.shape[-1]); t=(r[fit]/std).ravel(); weight=1/np.hypot(1,t/2.5)
    def report(e):
        return np.round([np.median(abs(e[:,:-1])),np.median(abs(e[:,-1])),np.max(np.abs(np.median(e[:,:-1],axis=0))),np.median(e[:,-1])],6).tolist()
    print('held lab',lab,'baseline (curve MAE median, rate MAE median, worst median horizon bias, rate median bias)',report(r[val]))
    for key,idx in [('saturation',slice(0,2)),('interactions',slice(2,None))]:
        coef=np.linalg.lstsq(a[:,idx]*weight[:,None],t*weight,rcond=1e-8)[0]
        print('   ',key,report(r[val]-alt[val][:,:,idx]@coef))
# Use only real calibration commands/readings to identify sensitivity to sparse panels.
c=d['calibration_load_cells']; labid=c['lab_id']; cmd=c['cmd_stress']; observed=c['true_stress']; gt=case['gt']; ref=np.column_stack((gt['stress_gains'],gt['stress_biases']))
for lab in range(4):
    ids=np.flatnonzero(labid==lab); envelope=np.r_[st[0][labs==lab].ravel(),st[1][labs==lab]]; lo,hi=(np.array([envelope.min(),envelope.max()])-cal[lab,1])/cal[lab,0]
    err={}
    for n in (2,3,4):
        for subset in combinations(ids,n):
            keep=labid!=lab; keep[list(subset)]=True
            for noise in (False,True):
                yy=observed.copy()
                if noise: yy[subset[0]]+=.6
                fit=s._calibrate(dict(lab_id=labid[keep],cmd_stress=cmd[keep],true_stress=yy[keep]))[lab]
                delta=fit-ref[lab]; value=max(abs(delta[0]*lo+delta[1]),abs(delta[0]*hi+delta[1])); key=(n,noise)
                if key not in err or value>err[key][0]: err[key]=(value,np.ptp(cmd[list(subset)]))
    print('lab',lab,'worst stress errors across spot subsets (n, EMI, max error, span)',[(n,noise,round(v,6),round(span,3)) for (n,noise),(v,span) in err.items()])
