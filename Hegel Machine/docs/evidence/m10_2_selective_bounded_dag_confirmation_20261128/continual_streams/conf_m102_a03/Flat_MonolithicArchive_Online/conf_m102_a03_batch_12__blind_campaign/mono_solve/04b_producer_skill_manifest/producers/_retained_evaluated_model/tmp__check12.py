import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task'); import batch_solver as b
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; f,m=b._operator(s['panel_b']); bias,g=b._calibration(s['panel_a']); z=(tr['x']-bias)/g; w=b._affine(np.cos(f*z))@m; x=b._affine(w); q=b._quadratic(w); p0=w[:,0]*w[:,2]; p1=w[:,1]*w[:,3]; scalar=[q,np.c_[x,p0,p1],np.c_[x,p0+p1]]; times=s['eval_times']; t=tr['traj']; y=tr['y']; n=len(y)
def basis(p): return -np.expm1(-np.outer(times,p))
def evaluate(i,j):
    yd=[]; td=[]
    for d in scalar: yd.append(np.abs(d[j]@b._fit(d[i],y[i])-y[j]))
    raw=b._fit(x[i],t[i]); modes=np.linalg.svd(x[i]@raw,full_matrices=False)[2][:2].T; td.append(np.abs((x[j]@raw@modes)@modes.T-t[j]))
    target=(np.linalg.qr(x[i])[0].T@t[i]).T
    def resid(p):
        a=basis(p); return (a@b._fit(a,target)-target).ravel()
    p=least_squares(resid,[.4,1.8],bounds=([.005,.6],[.8,15.])).x
    for rates in (p,[.42,1.83]):
        a=basis(rates); td.append(np.abs(x[j]@b._fit(x[i],b._fit(a,t[i].T).T)@a.T-t[j]))
    return np.array(yd),np.array(td),p

def report(name,i,j):
    e,t,p=evaluate(i,j); print(name,'y',e.mean(1),'state',t.mean((1,2)),'worst-time',t.mean(1).max(1),'max-y',e.max(1),'max-state',t.max((1,2)),'rates',p)
report('52/20',np.arange(52),np.arange(52,n)); rng=np.random.default_rng(718)
for k in range(8):
    perm=rng.permutation(n); report('random '+str(k),perm[:52],perm[52:])
for name,order in [('outer input',np.argsort(np.max(abs(z),1))),('large state coordinates',np.argsort(np.linalg.norm(w,axis=1)))]: report(name,order[:52],order[52:])
