import json, pickle
import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import least_squares
from itertools import product

F='/tmp/hegel_continual_archive/archive_calibration_history.json'
D=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_07.pkl','rb'))[0]['dataset']
N=len(D['train']['lab_id']); dt=D['past_dt']; t=D['eval_times']; K=len(t)
lab,grp,x,ref,pc,hc,s0=(np.concatenate((D['train'][k],D['test'][k])) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons','past_stress_cmd','stress_hold_cmd','strain_0'))
for c in range(4):
    a=lab==c; p=ref[a]; x[a]=(x[a]-np.nanmedian(np.nanmean(p,axis=0)))/np.nanmedian(np.nanstd(p,axis=0))
cell=np.array([[np.nanmean(x[(lab==c)&(grp==g)],axis=0) for g in range(3)] for c in range(4)])
xf={}
for mode in ('balanced','local'):
    f=np.nanmean(cell,axis=0)[grp] if mode=='balanced' else cell[lab,grp]
    xf[mode]=np.c_[np.ones(len(lab)),np.where(np.isfinite(x),x,f)]
yc=D['train']['strain_curves_obs']-s0[:N,None]; yr=D['train']['instant_rate_obs']; age=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]
arc=json.load(open(F)); h=arc[-1]; l,u,v=[np.asarray(h[k]) for k in ('lab_id','cmd_stress','true_stress')]
for c in [0,3]:
    print('recent suspected EMI',c,list(zip(u[l==c],v[l==c])))
cal=[(.79,-.14), tuple(np.polyfit(D['calibration_load_cells']['cmd_stress'],D['calibration_load_cells']['true_stress'],1)), (1.37,-.17), (1.24,-.13)]

def design(p,hold,features,act,lt,gate):
    tau=np.exp(lt); z=p@(np.exp(-age[:,None]/tau)*-np.expm1(-dt[:,None]/tau)); gap=hold[:,None]-z
    e=-np.expm1(-t[:,None]/tau); phi=np.log1p(t)[None,:] if not gate else np.where(act[:,None]>0,np.log1p(t),t)
    ah=act*hold; az=act*z[:,0]
    return np.concatenate((gap[:,None,:]*e, features[:,None,:]*phi[:,:,None], ah[:,None,None]*e[None,:,0,None], az[:,None,None]*phi[:,:,None]),axis=2), np.c_[gap/tau,features,ah/tau[0],az]

def run(cc,gate,mode,train,test,rob=True):
    g,b=np.asarray(cc)[lab].T; p=pc*g[:,None]+b[:,None]; hold=hc*g+b; damage=np.maximum(np.abs(p)-D['crit_stress'],0.)**2@dt; act=np.tanh(damage)
    y=np.r_[yc[train].ravel(),yr[train]]; m=len(train)*K
    def mat(lt,ids):
        ac,ar=design(p[ids],hold[ids],xf[mode][ids],act[ids],lt,gate)
        return np.r_[ac.reshape(-1,9),ar]
    w=np.r_[np.ones(m),np.full(len(train),2.)]
    def project(lt,get=False):
        a=mat(lt,train)*w[:,None]; b=y*w
        beta=lstsq(a,b,lapack_driver='gelsy',check_finite=False)[0]; e=a@beta-b
        return (e@e,np.r_[lt,beta]) if get else e
    lim=np.log([.025,80.]); fits=[]
    for a in [(0.24,2.2),(0.8,7.),(2.2,0.24),(7.,0.8)]:
        fit=least_squares(project,np.log(a),bounds=lim,max_nfev=45)
        fits.append(project(fit.x,True))
    _,params=min(fits,key=lambda q:q[0]); raw=lambda p: mat(p[:2],train)@p[2:]-y
    if rob:
        for _ in range(2):
            r=raw(params); med=lambda a:max(1e-6,1.4826*np.median(np.abs(a-np.median(a)))); scale=np.r_[np.full(m,med(r[:m])),np.full(len(train),med(r[m:]))]
            params=least_squares(lambda q:raw(q)/scale,params,bounds=(np.r_[[lim[0]]*2,[-np.inf]*9],np.r_[[lim[1]]*2,[np.inf]*9]),loss='soft_l1',f_scale=1.5,x_scale='jac',max_nfev=55).x
    ae=mat(params[:2],test)@params[2:]-np.r_[yc[test].ravel(),yr[test]]
    return ae[:len(test)*K].reshape(-1,K),ae[len(test)*K:],params,damage[test]

def report(e,r,p,damage,ids):
    def rm(v):return np.sqrt(np.mean(v*v)).round(5)
    print('all',rm(e),rm(r),'tau',np.exp(p[:2]).round(5),'hw',p[2:].round(5),'lab/active',[(str(c)+str(a),int(m.sum()),rm(e[m]),rm(r[m])) for c,a in product(range(4),[0,1]) if (m:=(lab[ids]==c)&((damage>0)==a)).any()])

if __name__=='__main__':
    ids=np.arange(N)
    for stress,gate,mode in product([(.85,.1),(1.37,-.17)],[False,True],['balanced','local']):
        cc=cal.copy(); cc[2]=stress
        print('candidate',stress,gate,mode); report(*run(cc,gate,mode,ids,ids),ids)
