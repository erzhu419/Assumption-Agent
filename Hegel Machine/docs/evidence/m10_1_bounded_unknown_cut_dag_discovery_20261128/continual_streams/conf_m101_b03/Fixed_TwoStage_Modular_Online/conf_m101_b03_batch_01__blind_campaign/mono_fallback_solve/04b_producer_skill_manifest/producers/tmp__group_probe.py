import importlib.util,pickle,numpy as np
from itertools import combinations, combinations_with_replacement
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
sp=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; z=m._inverse(tr['x'],m._calibrate(s['panel_a'])); ids=np.arange(len(z)); pairs=list(combinations(range(4),2))
def groups(z):
    gr=[z]; nonlinear=[]
    for a in z.T:
        g=[a*a,a**3,a**4]
        for k in (.5,1.,1.5,2.,2.5,3.,3.5,4.):
            g.extend((np.sin(k*a),np.cos(k*a),np.tanh(k*a),1/(1+k*a*a),a/(1+k*a*a),np.sqrt(1+k*a*a)))
        nonlinear.append(np.column_stack(g))
    gr.extend(nonlinear)
    for i,j in pairs:
        a,b=z[:,i],z[:,j]; w=a*b
        gr.append(np.column_stack([w]+[w*h for h in (a,b,a*a,b*b,w)]+[f(k*w) for k in (.7,1.5,2.5,3.5) for f in (np.sin,np.cos,np.tanh)]))
        gr.append(nonlinear[i]*b[:,None]);gr.append(nonlinear[j]*a[:,None])
    gr.append(np.column_stack([np.prod(z[:,p],axis=1) for n in (3,4) for p in combinations_with_replacement(range(4),n) if len(set(p))>=3]))
    return gr

def covariances(gs,a):
    k=[]
    for g in gs:
        f=(g-g[a].mean(0))/np.maximum(g[a].std(0),1e-12)
        k.append(f@f.T/f.shape[1])
    return np.array(k)

def predict(k,a,b,y,pen):
    ya=y[a]; ya=(ya-ya.mean())/ya.std(); ks=k[:,a][:,:,a]; ks=np.concatenate((ks,np.eye(len(a))[None])); flat=np.ascontiguousarray(ks.reshape(len(ks),-1)); prefer=np.r_[0,np.full(len(ks)-2,-3.),-8.]
    def objective(h):
        v=np.exp(h); c=cho_factor((v@flat).reshape(len(a),len(a))); w=cho_solve(c,ya); ci=cho_solve(c,np.eye(len(a))); diff=h-prefer
        return .5*ya@w+np.log(np.diag(c[0])).sum()+.5*pen*(diff@diff),.5*v*(flat@(ci-np.outer(w,w)).ravel())+pen*diff
    fit=minimize(objective,prefer,jac=True,method='L-BFGS-B',bounds=[(-13,5)]*(len(ks)-1)+[(-13,-3)],options={'maxiter':120}); w=np.exp(fit.x); c=cho_factor((w@flat).reshape(len(a),len(a)))
    return y[a].mean()+y[a].std()*np.einsum('k,kij,j->i',w[:-1],k[:,b][:,:,a],cho_solve(c,ya))

gs=groups(z); rng=np.random.RandomState(111); folds=[ids[52:],ids[56:],*np.array_split(rng.permutation(ids),4),*np.array_split(np.random.RandomState(927).permutation(ids),4)]
for b in folds:
    a=np.setdiff1d(ids,b); H,M=m._modes(s['eval_times'],tr['traj'][a]); targets=np.c_[tr['y'],tr['traj']@np.linalg.pinv(H.T)]; k=covariances(gs,a); diff=[]
    for pen in (.03,.15,.6):
        p=np.column_stack([predict(k,a,b,v,pen) for v in targets.T]); err=np.abs(p[:,1:]@H.T-tr['traj'][b]); diff.append([np.mean(abs(p[:,0]-tr['y'][b])),err.mean(),err.mean(0).max()]);
    print(len(b),np.round(diff,5).tolist(),flush=True)
