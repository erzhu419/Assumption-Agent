import sys
sys.path.insert(0,'/tmp')
from bridge_study import D,x,y,state,features,fit,inv
import numpy as np
from scipy.optimize import least_squares
pairs=((0,1),(1,2),(2,3),(0,3))
def calibrate(x,y,state):
    coefficients=[]
    for j,(i,k) in enumerate(pairs):
        s=np.column_stack([np.ones(len(y)),y,y[:,j]**3,y[:,i]*y[:,k],state])
        coefficients.append(np.linalg.lstsq(s,x[:,j],rcond=None)[0])
    return np.array(coefficients).T

def predict(y,w):
    return w[0]+y@w[1:5]+y**3*w[5]+np.stack([y[:,i]*y[:,j] for i,j in pairs],axis=1)*w[6]

def recover(x,w):
    z=(x-w[0])@np.linalg.inv(w[1:5]); z=np.clip(z,-1.8,1.8)
    for _ in range(30):
        jac=np.broadcast_to(w[1:5].T,(len(z),4,4)).copy()
        for j,(i,k) in enumerate(pairs):
            jac[:,j,j]+=3*w[5,j]*z[:,j]**2
            jac[:,j,i]+=w[6,j]*z[:,k]
            jac[:,j,k]+=w[6,j]*z[:,i]
        step=np.linalg.solve(jac,(x-predict(z,w))[...,None])[...,0]
        z+=np.clip(step,-.5,.5)
        if np.max(abs(step))<1e-10:break
    return z

if __name__=='__main__':
    rng=np.random.default_rng(14)
    random=np.array_split(rng.permutation(len(y)),5)
    balanced=[np.r_[a,b] for a,b in zip(np.array_split(rng.permutation(np.flatnonzero(~state)),3),np.array_split(rng.permutation(np.flatnonzero(state)),3))]
    for label,folds in [('random',random),('state-balanced',balanced)]:
        for f in folds:
            tr=np.setdiff1d(np.arange(len(y)),f); w=calibrate(x[tr],y[tr],state[tr]); z=recover(x[f]-state[f,None]*w[-1],w); e=z-y[f]
            print(label,'shifted train/test',state[tr].sum(),state[f].sum(),'total/channel RMSE',np.sqrt(np.mean(e.sum(1)**2)),np.sqrt(np.mean(e**2)),'worst row',f[np.argmax(np.linalg.norm(e,axis=1))])
    w=calibrate(x,y,state)
    print('coefficients',w)
    for g in ('train','test'):
        for key in ('past_channel_raw','hold_channel_raw'):
            raw=D[g][key].reshape(-1,4); z=recover(raw,w); print(g,key,'worst closure',np.max(abs(predict(z,w)-raw)),'load extent',z.min(),z.max())
    import importlib.util
    spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    tr=D['train']; t=D['eval_times']; dt=D['past_dt']; a=m._aligned_features(tr,D['test'])[:len(tr['lab_id'])]; c=(recover(tr['past_channel_raw'].reshape(-1,4),w).sum(1).reshape(tr['past_channel_raw'].shape[:-1]),recover(tr['hold_channel_raw'],w).sum(1),a); Y=tr['strain_curves_obs']-tr['strain_0'][:,None]; R=tr['instant_rate_obs']; tau,beta=m._fit(dt,t,c,Y,R); print('full intrinsic',tau,beta)
    for lab in np.unique(tr['lab_id']):
        mask=tr['lab_id']==lab; tau,beta=m._fit(dt,t,tuple(a[~mask] for a in c),Y[~mask],R[~mask]); A,B=m._design(tau,dt,t,tuple(a[mask] for a in c)); e=abs(A@beta-Y[mask]); er=abs(B@beta-R[mask]); print('held lab',lab,'taus',tau,'errors',np.max(np.median(e,axis=0)),np.median(er),np.quantile(e,.9),np.quantile(er,.9))
