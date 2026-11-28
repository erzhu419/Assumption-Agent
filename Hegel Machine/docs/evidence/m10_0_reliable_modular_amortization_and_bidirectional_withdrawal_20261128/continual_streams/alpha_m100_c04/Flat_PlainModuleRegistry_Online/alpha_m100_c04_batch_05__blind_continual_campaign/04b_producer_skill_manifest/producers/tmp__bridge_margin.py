import pickle,sys,itertools
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
import upstream_transducer as u
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; X=C['true_channel_loads']; Y=C['bridge_raw']; S=C['single_channel_sweeps']; good=np.max(abs(Y),axis=1)<5; axis=np.count_nonzero(X,axis=1)<=1
U=u._OPPOSITE
print('Relevant dataset metadata:', [(k,type(v).__name__) for k,v in D.items() if k not in ('train','test')]); print('train/test fields:', list(D['train']),list(D['test']))

def fit(c, family):
    x=c['true_channel_loads']; y=c['bridge_raw']; s=c['single_channel_sweeps']; n=len(x)
    ix=[s['lab_id']==j for j in range(4)]
    t=np.array([np.r_[x[:,j],s['true_stress'][m]] for j,m in enumerate(ix)])
    v=np.array([np.r_[x[:,k],np.zeros(m.sum())] for k,m in zip(U,ix)])
    z=np.array([np.r_[y[:,j],s['cmd_stress'][m]] for j,m in enumerate(ix)])
    def shape(t,v,p):
        f,h,b,a,d=p[:5]; extra=0
        if family=='ridge': v=v+p[5]*t
        if family=='rational': extra=p[5]*(t+v)**2/(1+(t-v)**2)
        if family=='product': extra=p[5]*t*v
        return t+b*v+a*np.sin(f*t)+d*(np.cos(h*v)-1)+extra
    def profile(p,w):
        g=shape(t,v,p); sw=w.sum(axis=1); gm=(w*g).sum(axis=1)/sw; zm=(w*z).sum(axis=1)/sw
        gc=g-gm[:,None]; gain=(w*gc*(z-zm[:,None])).sum(axis=1)/(w*gc**2).sum(axis=1); bias=zm-gain*gm
        return bias,gain,bias[:,None]+gain[:,None]*g-z
    p=[2.5,2.,.22,.25,.06]+([] if family=='shared' else [0.]); w=np.where(abs(z)<5,1.,1e-12); w[:,:n]=np.where(np.max(abs(y),axis=1)<5,1.,1e-12)
    for _ in range(3):
        p=least_squares(lambda p:(profile(p,w)[2]*np.sqrt(w)).ravel(),p).x
        bias,gain,e=profile(p,w); scale=max(1e-7,1.4826*np.median(abs(e-np.median(e)))); a=abs(e); a[:,:n]=np.max(a[:,:n],axis=0); w=1/(1+(a/(8*scale))**2)
    coeff=np.column_stack((bias,gain,gain*p[2],gain*p[3],gain*p[4])); return (coeff,p[:2]),p,lambda x:bias+gain*shape(x,x[...,U],p)

base=u._fit(C); print('independent normalized coefficients:',base[0][:,2:]/base[0][:,1:2]); families=['independent','shared','ridge','rational','product']; rng=np.random.default_rng(131)
folds=[np.r_[a,b] for a,b in zip(np.array_split(rng.permutation(np.flatnonzero(good&axis)),8),np.array_split(rng.permutation(np.flatnonzero(good&~axis)),8))]; stress=np.unique(S['true_stress']);
for name in families:
    res=np.empty_like(Y); inv=np.empty_like(X); swerr=[]; swlab=[]
    for j,idx in enumerate(folds):
        m=~np.isin(np.arange(len(X)),idx); s=~np.isin(S['true_stress'],stress[j::len(folds)]); c={**C,'true_channel_loads':X[m],'bridge_raw':Y[m],'single_channel_sweeps':{k:v[s] for k,v in S.items()}}
        if name=='independent':
            model=u._fit(c); forward=lambda x:u._forward(x,*model)
        else: model,p,forward=fit(c,name)
        res[idx]=forward(X[idx])-Y[idx]
        if name in ('independent','shared'): inv[idx]=u._invert(Y[idx],*model)-X[idx]
        s=(~s)&(abs(S['cmd_stress'])<5); lab=S['lab_id'][s]; swlab.extend(lab); swerr.extend((forward(np.eye(4)[lab]*S['true_stress'][s,None])[np.arange(len(lab)),lab]-S['cmd_stress'][s]).tolist())
    def stats(a): return np.round([np.sqrt(np.mean(a*a)),np.max(abs(a))],8).tolist()
    print(name,'forward axis/mixed/sweep RMSE,max:',*(stats(a) for a in (res[good&axis],res[good&~axis],np.array(swerr))),flush=True)
    if name in ('independent','shared'): print(' inverse axis/mixed/challenging mixed:',*(stats(inv[m]) for m in (good&axis,good&~axis,good&~axis&(np.max(abs(X),axis=1)>1.15))),flush=True)
    if name!='independent': print(' shape:',np.round(fit(C,name)[1],8),flush=True)
