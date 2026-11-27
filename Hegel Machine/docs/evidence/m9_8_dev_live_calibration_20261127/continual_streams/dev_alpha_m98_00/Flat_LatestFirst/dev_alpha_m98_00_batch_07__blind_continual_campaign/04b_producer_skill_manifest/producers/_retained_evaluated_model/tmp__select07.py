import contextlib,io,runpy,numpy as np
from scipy.optimize import least_squares
with contextlib.redirect_stdout(io.StringIO()):
    q=runpy.run_path('/tmp/probe07.py')
g=q['g']; H,Z,tau,b,t,G,C,Y,R,X=(q[k] for k in ('H','Z','tau','b','t','G','C','Y','R','Xall')); X=X['diag']; drift=X@b[2:]; amp=(H[:,None]-Z)*b[:2]; Y0=amp@G.T+drift[:,None]*t; R0=(amp/tau).sum(1)+drift; e=Y-Y0; f=R-R0
arch=np.array([0.41228893817869877,2.2768586984955768]); rm=lambda x:np.sqrt(np.mean(x*x)); print('diagnostics standalone, rate by lab and stratum',[[round(rm(f[g[k]==v]),6) for v in np.unique(g[k])] for k in ('lab_id','stratum')]); print('archive forcing',arch,'relative to tau',arch/tau)
for model in ('archive','common_ratio','two_forcing','two_plus_drift','two_plus_memory'):
    M=drift[:,None,None]*C[None,:,:]; N=np.zeros((len(H),2)); offset=np.zeros_like(e)
    if model=='archive': offset=M@arch; M=M[:,:,:0]; N=N[:,:0]
    if model=='common_ratio': M=(M@tau)[:,:,None]; N=N[:,:1]
    if model=='two_plus_drift': M=np.concatenate([M,drift[:,None,None]*t[None,:,None]],axis=2); N=np.c_[N,drift]
    if model=='two_plus_memory': M=np.concatenate([M,Z[:,None,:]*G[None,:,:]],axis=2); N=np.c_[N,Z/tau]
    def fit(s):
        if M.shape[-1]==0: return np.array([])
        return least_squares(lambda p:np.r_[((M[s]@p+offset[s]-e[s])/.007).ravel(),(N[s]@p-f[s])/.007],np.zeros(M.shape[-1]),loss='soft_l1').x
    p=fit(np.ones(len(H),dtype=bool)); yh=M@p+offset; rh=N@p; print(model,'full coefficients',p.tolist(),'curve/rate',rm(e-yh),rm(f-rh));
    for key in ('lab_id','stratum'):
        vals=[]
        for h in np.unique(g[key]):
            s=g[key]==h; a=fit(~s); err=e[s]-M[s]@a-offset[s]; er=f[s]-N[s]@a; vals.append([rm(err),rm(er),rm(err[:,:4]),rm(err[:,-4:])])
        print(key,'worst curve/rate/early/late',np.max(vals,axis=0).round(6))
