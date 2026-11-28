import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import solve_batch,_design,_geometry,_ls,_time_fit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y,S=tr['y'],tr['traj']; t=s['eval_times']; H=_design(*_geometry(s,tr['x'])); n=len(y)
def fit(i,j,model):
    Y=np.column_stack((y,S)); A=_ls(H[i],Y[i]); yhat=H[j]@A[:,0]; P=H
    if model.startswith('shared'):
        # Use the endpoint to identify the common spatial contribution, without using held-out endpoints.
        cut=int(model.split(':')[1]); P=np.column_stack((H[:,:cut],H[:,cut:]@A[cut:,0])); A=_ls(P[i],S[i]);
    else: A=A[:,1:]
    pred=P[j]@A
    if model!='direct':
        if model=='two':
            from scipy.optimize import least_squares
            basis=lambda k:-np.expm1(-t[:,None]*k)
            Q=P[i]@A
            def err(k):
                B=basis(k); return (B@_ls(B,Q.T)-Q.T).ravel()
            B=basis(least_squares(err,[.45,2.5],bounds=(.01,15)).x)
        else: B=_time_fit(t,S[i],P[i]@A)
        pred=(_ls(B,pred.T).T)@B.T
    return yhat,pred
rng=np.random.default_rng(42); splits={'56/16':np.arange(56,n)}
for k in range(4): splits['random'+str(k)]=rng.permutation(n)[:16]
z,*_=_geometry(s,tr['x']); r=np.linalg.norm(z,axis=1); splits['outer radius']=np.argsort(r)[-16:]; splits['inner radius']=np.argsort(r)[:16]
for model in ['direct','two','three','shared:11','shared:15','shared:21']:
    metrics=[]
    for key,j in splits.items():
        i=np.setdiff1d(np.arange(n),j); a,b=fit(i,j,model); err=abs(b-S[j]); metrics.append([np.mean(abs(a-y[j])),err.mean(),err.mean(0).max(),err[:,-1].mean()]);
        if model=='three':
            sub=dict(s,train={k:(v[i] if k in ('x','y','traj') else v) for k,v in tr.items()},test={'x':tr['x'][j]}); q=solve_batch(sub); assert np.allclose(q['y_hat'],a) and np.allclose(q['state_pred'],b); e=b[:,-1]; assert np.allclose(q['diagnostics_vec'],[a.mean(),a.std(),e.mean(),e.std()])
    m=np.array(metrics); print(model,'56/16',m[0].round(6),'mean',m.mean(0).round(6),'worst',m.max(0).round(6),'worst slices',[list(splits)[k] for k in m.argmax(0)])
