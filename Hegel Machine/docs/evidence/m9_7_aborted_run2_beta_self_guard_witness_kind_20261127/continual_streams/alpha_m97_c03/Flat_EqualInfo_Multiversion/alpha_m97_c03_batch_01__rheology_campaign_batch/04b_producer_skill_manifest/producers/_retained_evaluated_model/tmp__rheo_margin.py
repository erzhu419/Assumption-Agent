import pickle
from pathlib import Path
import numpy as np
p=Path('/holdout_task/alpha_m97_c03_batch_01_solver.py')
source=p.read_text()
models={}
for name, code in [('current',source),('cauchy',source.replace("loss='soft_l1', f_scale=2.","loss='cauchy', f_scale=2.5")),('cauchy4',source.replace("loss='soft_l1', f_scale=2.","loss='cauchy', f_scale=4."))]:
    env={};exec(compile(code,str(p),'exec'),env);models[name]=env
items=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_01.pkl','rb')); print('public cases',len(items))
for item in items:
    d=item['dataset']; tr,te=d['train'],d['test']; lab=tr['lab_id']; complete=np.isfinite(tr['coupon_features_raw']).all(axis=1)
    s=models['current']; g,b=s['_calibrate'](d['calibration_load_cells']); x,_=s['_align'](tr,te); dt,t=d['past_dt'],d['eval_times']; args=g,b,dt,t,d['crit_stress'],d['shear_banding_index']; f,active=s['_design'](tr,x,*args)
    y=tr['strain_curves_obs']-tr['strain_0'][:,None]; rate=tr['instant_rate_obs']; rmse=lambda e:np.sqrt(np.nanmean(e**2)); gt=item['gt']; cols=('taus','theta','vp_theta','te_clean','te_true_rate0')
    for name,s in models.items():
        slices=[]; bias=[]
        for c in np.unique(lab):
            keep=(lab!=c)&complete
            lt,w=s['_fit'](f,np.where(keep[:,None],y,np.nan),np.where(keep,rate,np.nan),dt,t)
            C,R=f(lt); e=C@w-y; re=R@w-rate
            for a in (False,True):
                for full in (True,False):
                    m=(lab==c)&(active==a)&(complete==full)
                    if m.any(): slices.append((c,a,full,np.nanmedian(abs(e[m])),np.nanquantile(abs(e[m]),.90),np.nanquantile(abs(e[m]),.95),rmse(re[m])))
            m=(lab==c)&complete; bias.append((int(c),round(float(np.nanmedian(re[m])),5),round(float(np.nanmedian(e[m])),5)))
        out=s['solve_alpha_m97_c03_batch_01'](d)
        print(name,'public RMSE',np.round([rmse(z-gt[k]) for z,k in zip(out.values(),cols)],6),'held-lab complete and missing maxima [median,p90,p95,rateRMSE]',*[np.round(np.max([z[3:] for z in slices if z[2]==v],axis=0),5) for v in (True,False)],'complete held-lab signed median [lab,rate,strain]',bias)
