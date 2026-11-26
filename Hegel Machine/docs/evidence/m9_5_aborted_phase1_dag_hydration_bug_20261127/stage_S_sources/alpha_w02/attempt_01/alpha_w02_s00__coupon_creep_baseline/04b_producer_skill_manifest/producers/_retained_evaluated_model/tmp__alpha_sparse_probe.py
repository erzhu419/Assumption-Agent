import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
root='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',root+'alpha_w02_s00_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
a=pickle.load(open(root+'public_alpha_w02_s00.pkl','rb'))[0]; d,g=a['dataset'],a['gt']; tr=d['train']; checks=d['calibration_load_cells']
lab=tr['lab_id']; dt,t=d['past_dt'],d['eval_times']; x,_,complete=s._align(d); hist,hold=s._state_inputs(tr,s._calibration(d)); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']
true=np.column_stack((g['stress_gains'],g['stress_biases']))
for l in range(4):
    m=(lab!=l)&complete; j=(lab==l)&complete
    tau,w=s._fit(hist[m],hold[m],dt,t,x[m],y[m],r[m]); h=tr['past_stress_cmd'][j]; p=tr['stress_hold_cmd'][j]
    bc,br=s._basis(tau,h,p,dt,t,x[j]); ac,ar=bc[:,:,:2]@w[:2],br[:,:2]@w[:2]
    bc,br=s._basis(tau,np.ones_like(h),np.ones_like(p),dt,t,x[j]); ec,er=bc[:,:,:2]@w[:2],br[:,:2]@w[:2]
    drift=w[2]+x[j]@w[3:]
    # Profile a lab drift offset/scale so finite-panel alignment error does not look like bias or gain.
    C=np.column_stack((ac.ravel(),ec.ravel(),np.tile(t,j.sum()),(drift[:,None]*t).ravel()))
    R=np.column_stack((ar,er,np.ones(j.sum()),drift))
    design=np.vstack((C/.025,R/.014)); yy=np.r_[((y[j]-drift[:,None]*t)/.025).ravel(),(r[j]-drift)/.014]
    z=least_squares(lambda z: design@z-yy,[1.,0.,0.,0.],loss='cauchy',f_scale=2).x
    mc=checks['lab_id']==l; q=checks['cmd_stress'][mc]; v=checks['true_stress'][mc]; err=[]; base=[]
    for qi,vi in zip(q,v):
        A=design[:,[0,2,3]].copy(); A[:,0]-=qi*design[:,1]; Y=yy-vi*design[:,1]
        p=least_squares(lambda p: A@p-Y,z[[0,2,3]],loss='cauchy',f_scale=2).x
        err.append(np.abs(p[0]-true[l,0])*2.5+abs(vi-qi*p[0]-true[l,1])); base.append(abs(1-true[l,0])*2.5+abs(vi-qi-true[l,1]))
    print(l,'true',true[l].round(5),'response-only gain,bias',z[:2].round(5),'response/check residual max',round(np.max(np.abs(v-(z[0]*q+z[1]))),5),'single-check max error baseline -> anchored',round(max(base),5),round(max(err),5),'two-stress design svals',np.linalg.svd(design,compute_uv=False).round(2))
