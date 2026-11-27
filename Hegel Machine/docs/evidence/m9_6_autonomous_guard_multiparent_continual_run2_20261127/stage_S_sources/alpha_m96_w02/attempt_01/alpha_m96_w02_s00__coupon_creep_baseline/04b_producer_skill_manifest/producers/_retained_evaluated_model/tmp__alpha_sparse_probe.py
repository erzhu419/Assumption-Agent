import importlib.util, pickle, numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m96_w02_s00_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
c=pickle.load(open('/holdout_task/public_alpha_m96_w02_s00.pkl','rb'))[0]; d,g=c['dataset'],c['gt']; tr,te=d['train'],d['test']

def joint(d):
    tr,te=d['train'],d['test']; lab=tr['lab_id']; checks=d['calibration_load_cells']; cl=checks['lab_id']; cmd=checks['cmd_stress']; val=checks['true_stress']; cal=s._stress_calibration(checks)
    sparse=[]; anchors=[]
    for j in range(4):
        m=(cl==j)&np.isfinite(cmd)&np.isfinite(val)
        if m.any() and np.ptp(cmd[m])<1e-8 and np.any(lab==j):
            sparse.append(j); anchors.append(np.median(cmd[m])); cal[j]=[1.,np.median(val[m]-cmd[m])]
    x=s._align(tr,te); n=len(lab); dt,t=d['past_dt'],d['eval_times']; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; rate=tr['instant_rate_obs']; fit=~np.isin(lab,sparse)
    if not fit.any(): return s.solve_alpha_m96_w02_s00(d),cal
    base=s._inputs(tr,x[:n],cal); base_design=s._Design(dt,t,base)
    pilot=s._Design(dt,t,tuple(a[fit] for a in base)); tau,theta=s._fit(pilot,y[fit],rate[fit]); target=np.column_stack((y,rate)); r=(base_design.make(tau)@theta-target)[fit]
    def mad(a): return 1.4826*np.median(abs(a-np.median(a)))
    scale=2.5*np.r_[np.full(len(t),max(1e-7,mad(r[:,:-1]))),max(1e-7,mad(r[:,-1]))]
    # At the known check (x*,y*), sigma(x)=y*+gain*(x-x*).
    # Design each lab's exact derivative with respect to gain.
    contrasts=[]
    for j,a in zip(sparse,anchors):
        m=lab==j
        contrasts.append(s._Design(dt,t,(tr['past_stress_cmd'][m]-a,tr['stress_hold_cmd'][m]-a,base[2][m])))
    def error(p):
        a=base_design.make(p[:2]);
        for k,(j,contrast) in enumerate(zip(sparse,contrasts)):
            a[lab==j,:,:2]+=(np.exp(p[9+k])-1)*contrast.make(p[:2])[:,:,:2]
        return ((a@p[2:9]-target)/scale).ravel()
    p=least_squares(error,np.r_[tau,theta,np.zeros(len(sparse))],bounds=(np.r_[.2,.2,np.full(7+len(sparse),-np.inf)],np.r_[8.,8.,np.full(7+len(sparse),np.inf)]),loss='cauchy',x_scale='jac',max_nfev=150,ftol=1e-9).x
    for j,a,q in zip(sparse,anchors,p[9:]):
        gain=np.exp(q); cal[j,1]-=(gain-1)*a; cal[j,0]=gain
    f=s._Design(dt,t,s._inputs(te,x[n:],cal)).make(p[:2])@p[2:9]
    if 'instant_rate_obs' in te:
        onset=te['instant_rate_obs']; m=np.isnan(te['coupon_features_raw']).any(axis=1)&np.isfinite(onset); f[m,:-1]+=(onset[m]-f[m,-1])[:,None]*t; f[m,-1]=onset[m]
    return dict(predicted_strain=f[:,:-1]+te['strain_0'][:,None],initial_rate_pred=f[:,-1],estimated_taus=p[:2],hereditary_weights=p[2:9]),cal

def recode(d, pair, spot):
    q={k:v.copy() for k,v in d['calibration_load_cells'].items()}; recoded={}
    for name in ('train','test'):
        b=d[name]; recoded[name]=dict(b,past_stress_cmd=b['past_stress_cmd'].copy(),stress_hold_cmd=b['stress_hold_cmd'].copy())
    mask=np.ones(len(q['lab_id']),bool)
    for lab,a,bias in pair:
        m=q['lab_id']==lab; q['cmd_stress'][m]=(q['cmd_stress'][m]-bias)/a
        for z in recoded.values():
            sel=z['lab_id']==lab
            for k in ('past_stress_cmd','stress_hold_cmd'): z[k][sel]=(z[k][sel]-bias)/a
        idx=np.flatnonzero(m); idx=idx[np.argsort(q['cmd_stress'][idx])]; mask[idx]=False; mask[idx[spot]]=True
    return dict(d,**recoded,calibration_load_cells={k:v[mask] for k,v in q.items()})

if __name__=='__main__':
    rmse=lambda x: np.sqrt(np.mean(x*x)); wing=te['lab_id']>=2
    for pair in [[(2,1.,0.),(3,1.,0.)],[(2,1.12,.06),(3,.91,-.04)],[(2,.80,-.03),(3,1.25,.07)]]:
        for spot in (0,5,-1):
            q=recode(d,pair,spot); old=s.solve_alpha_m96_w02_s00(q); new,cal=joint(q)
            errors=lambda o,m: (rmse(o['predicted_strain'][m]-g['te_clean'][m]),rmse(o['initial_rate_pred'][m]-g['te_true_rate0'][m]))
            print('gain',tuple(a for j,a,b in pair),'spot',spot,'wing baseline',errors(old,wing),'joint',errors(new,wing),'anchor wing',errors(new,~wing),'estimated gains',cal[2:,0],flush=True)
    pair=[(2,1.12,.06),(3,.91,-.04)]
    for fold in range(3):
        hold=(np.arange(len(tr['lab_id']))//4)%3==fold
        ds=dict(d,train={k:v[~hold] for k,v in tr.items()},test={k:v[hold] for k,v in tr.items() if k!='strain_curves_obs'})
        ds=recode(ds,pair,5); o=s.solve_alpha_m96_w02_s00(ds); z,cal=joint(ds); m=ds['test']['lab_id']>=2
        print('within-lab fold',fold,'wing B strain MdAE',[np.median(abs(v['predicted_strain'][m]-tr['strain_curves_obs'][hold][m])) for v in (o,z)],'gain',cal[2:,0],flush=True)
