import pickle, numpy as np
from itertools import combinations, combinations_with_replacement
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; x,y=C['bridge_raw'],C['true_channel_loads']; np.set_printoptions(precision=6,suppress=True,linewidth=150)
print('largest raw norms',sorted(np.max(abs(x),axis=1))[-8:]); state=np.max(abs(x),axis=1)>8; print('offset rows',np.flatnonzero(state))
def features(y,kind):
    out=[np.ones(len(y)),*y.T]
    if kind in ('diag2','diag3','diag5','diag3_pair','full3','full2'):out.extend((y*y).T)
    if kind in ('odd3','diag3','diag5','diag3_pair','full3'):out.extend((y**3).T)
    if kind=='diag5':out.extend((y**4).T);out.extend((y**5).T)
    if kind in ('diag3_pair','full3','full2'):out.extend(y[:,i]*y[:,j] for i,j in combinations(range(4),2))
    if kind=='full3':out.extend(np.prod(y[:,ij],axis=1) for ij in combinations_with_replacement(range(4),3) if len(set(ij))>1)
    return np.column_stack(out)
def fit(y,x,state,kind):
    a=features(y,kind); use=state.any(); w=np.linalg.lstsq(np.column_stack([a,state]) if use else a,x,rcond=None)[0]; return (w[:-1],w[-1]) if use else (w,np.zeros(4))
def inv(target,w,kind):
    z=(target-w[0])@np.linalg.inv(w[1:5]); z=np.clip(z,-2,2)
    for i in range(20):
        e=target-features(z,kind)@w; j=np.stack([(features(z+v,kind)-features(z-v,kind))@w/2e-5 for v in np.eye(4)*1e-5],axis=2)
        step=np.linalg.solve(j,e[...,None])[...,0]; z+=np.clip(step,-.5,.5)
        if np.max(abs(step))<1e-9:break
    return z
if __name__=='__main__':
 rng=np.random.default_rng(14); folds=np.array_split(rng.permutation(len(y)),5)
 for kind in ('lin','diag2','odd3','diag3','diag5','full2','diag3_pair','full3'):
    w,o=fit(y,x,state,kind); forward=x-features(y,kind)@w-state[:,None]*o
    cv=[]; indiv=[]
    for f in folds:
        tr=np.setdiff1d(np.arange(len(y)),f); q,s=fit(y[tr],x[tr],state[tr],kind); de=inv(x[f]-state[f,None]*s,q,kind)-y[f]; cv.append(np.sqrt(np.mean(de.sum(1)**2)));indiv.append(np.sqrt(np.mean(de**2)))
    print(kind,'offset',o,'forward RMSE',np.sqrt(np.mean(forward**2)),'worst CV total/channel',max(cv),max(indiv))
    if kind in ('odd3','diag3','diag3_pair'): print('coeffs',w)
