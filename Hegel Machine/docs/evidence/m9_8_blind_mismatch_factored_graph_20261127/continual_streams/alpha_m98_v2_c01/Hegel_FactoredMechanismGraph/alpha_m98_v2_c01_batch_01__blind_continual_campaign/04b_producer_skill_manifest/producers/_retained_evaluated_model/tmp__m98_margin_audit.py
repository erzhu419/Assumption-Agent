import importlib.util, pickle
import numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; g=p['campaign_diagnostics']; result=m.solve_batch(d)
t=d['eval_times']; tau=result['estimated_taus']; theta=result['hereditary_weights']; cal=d['calibration_load_cells']; bridge=m._axial_load_map(cal)
cp=cal['true_channel_loads'].sum(1); raw=cal['bridge_raw']; err=np.empty_like(cp)
# Hold out calibration load levels rather than measuring fit error on the same rows.
for block in np.array_split(np.argsort(cp),5):
    keep=~np.isin(np.arange(len(cp)),block)
    a,b=m._axial_load_map({'bridge_raw':raw[keep],'true_channel_loads':cal['true_channel_loads'][keep]}); err[block]=(raw[block]-a)@b-cp[block]
    print('bridge held-out load interval', np.round([cp[block].min(),cp[block].max()],4), 'RMSE',np.sqrt(np.mean(err[block]**2)))
lab=g['lab_id']; groups=np.unique(lab); x=np.empty_like(g['coupon_features_raw'])
for l in groups:
    k=lab==l; ref=g['ref_coupons'][k]; x[k]=(g['coupon_features_raw'][k]-np.median(ref.mean(0)))/np.median(ref.std(0))
# The diagnostics use their own observed reference panel. This does not change production preprocessing.
missing=np.isnan(x).any(1); past,hold,_=m._specimen(g,x,bridge)
_, r=m._branches((past,hold,None),d['past_dt'],t,tau); mem=hold[:,None]-r*tau
B=np.vstack((-np.expm1(-t[:,None]/tau),1/tau)); y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]-hold[:,None]*(B@theta[:2]); w=np.r_[np.ones(len(t)),.65]
options=[('reset',0),('past pair difference',1),('slow memory',2),('hold–drift interaction',3),('rational drift gate',4,.5),('rational drift gate',4,2.),('secular drift',5)]
def design(mask, opt):
    X=np.where(np.isnan(x),np.nanmean(x[mask],axis=0),x); D=theta[2]+X@theta[3:]; mode=opt[1]
    cols=D[:,None,None]*B[None,:,:]
    if mode==0:return cols
    if mode==5:return np.concatenate([cols,D[:,None,None]*np.r_[t,1][None,:,None]],axis=2)
    H=hold-np.mean(hold[mask]); q=[None,past[:,-1]-past[:,-2],mem[:,1],H*D][mode] if mode<4 else D/(1+opt[2]*H**2)-D
    return np.concatenate([cols,q[:,None,None]*B[None,:,:]],axis=2)
def fit(train,val,opt):
    A=design(train,opt); z=(A[train]*w[None,:,None]).reshape(-1,A.shape[-1]); b=np.linalg.lstsq(z,(y[train]*w).ravel(),rcond=None)[0]; return A[val]@b-y[val]
def scores(e):
    rm=lambda z: np.sqrt(np.mean(z*z)); return np.round([rm(e[:,:-1]),rm(e[:,-1]),max(rm(e[lab==l,:-1]) for l in groups),max(rm(e[lab==l,-1]) for l in groups),rm(e[missing]),rm(e[~missing])],6)
base=np.empty_like(y); challenge=np.empty_like(y); shipped=np.empty_like(y)
for l in groups:
    val=lab==l; train=~val
    def inner(opt):
        return max(np.sqrt(np.mean(fit(train&(lab!=j),lab==j,opt)**2)) for j in groups if j!=l)
    opt=min(options,key=inner)
    base[val]=fit(train,val,options[0]); challenge[val]=fit(train,val,opt)
    shipped[val]=design(train,options[0])[val]@np.array([.34308,2.13589])-y[val]
    print('held-out lab',l,'inner winner',opt,'reset mean curve/rate residual',np.mean(base[val,:-1]),np.mean(base[val,-1]))
print('columns: curve RMSE, rate RMSE, worst-lab curve RMSE, worst-lab rate RMSE, missing-row joint RMSE, complete-row joint RMSE')
print('reset OOF',scores(base)); print('nested structural selection OOF',scores(challenge)); print('fixed shipped gains diagnostic replay (not OOF)',scores(shipped))
