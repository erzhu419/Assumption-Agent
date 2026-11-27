import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=m._Transport(s); decode=m._calibration(s); b=m._pulse_losses(t,s['reference_tracer_pulses'],decode); Z=s['Z_nuisance']; y=decode(s['y_obs_raw']); S0,U0,_=t.response(b,t.sources)
cap=s['u_box']; budget=s['total_budget']; w0,_=m._source_fit(s,S0,y); resid=y-S0@w0; G=np.repeat(np.eye(2),2,axis=0); banks=np.asarray(s['row_bank_id']); groups=np.tile(np.arange(len(t.sensors))%4,len(t.shifts)); print('Measured endpoint losses minus nominal:',b-t.nominal)
configs=[(0.0,0.0),(0.002,0.002),(0.01,0.01),(0.04,0.04),(0.0,0.01)]
errors={k:[] for k in ['nominal',*configs]}; sensors={k:[] for k in errors}; real=[]
for scale in [None,-2,-1,0,1,2]:
    truth=b if scale is None else t.nominal+scale*(b-t.nominal); S,U,_=t.response(truth,t.sources); obs=y if scale is None else S@w0+resid
    target_w,_=m._source_fit(s,S,obs); target=U@target_w
    for kind in ['nominal',*configs]:
        out=[]
        for test in [np.zeros(len(y),bool),*[groups==g for g in range(4)]]:
            train=~test; zz=Z[train]; P=np.linalg.pinv(zz); project=lambda v:v[train]-zz@(P@v[train]); yp=project(obs)
            def profile(q,return_state=False):
                boundary=t.nominal+G@q; X,V,_=t.response(boundary,t.sources); Xp=project(X)
                rig,pen=(0.002,0.) if kind=='nominal' else kind; H=Xp.T@Xp+(rig+1e-9)*np.eye(X.shape[1]); w,_=m._box_simplex(H,Xp.T@yp,np.full(X.shape[1],cap),budget)
                if return_state:return X,V,w,boundary
                return np.r_[Xp@w-yp,np.sqrt(rig)*(w-budget/len(w)),np.sqrt(pen)*q]
            q=np.zeros(2) if kind=='nominal' else least_squares(profile,np.zeros(2),bounds=(-2,2),gtol=1e-9).x
            X,V,w,loss=profile(q,True); r=obs-X@w; gamma=P@r[train]
            if np.any(test):out.append(np.sqrt(np.mean((r[test]-Z[test]@gamma)**2)))
            else:
                w,_=m._source_fit(s,X,obs); error=(np.max(abs(loss-truth)),np.linalg.norm(V@w-target)/np.linalg.norm(target),np.max(abs(w-target_w))); errors[kind].append(error); raw=(q,error)
        sensors[kind].append(max(out))
        if scale is None: print('real',kind,'correction',raw[0].round(4),'loss/field/weight error',np.round(raw[1],6),'worst held-out RMSE',max(out))
for k in errors:print('perturbed real residual',k,'worst loss/field/weight error',np.max(errors[k][1:],axis=0),'worst held-out RMSE',max(sensors[k][1:]))
