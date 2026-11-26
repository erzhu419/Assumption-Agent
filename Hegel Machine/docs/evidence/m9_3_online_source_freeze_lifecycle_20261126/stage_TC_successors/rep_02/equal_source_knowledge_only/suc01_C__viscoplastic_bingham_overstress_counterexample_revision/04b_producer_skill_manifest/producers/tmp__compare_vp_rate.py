exec(open('/tmp/stress_vp.py').read().split('for i, case in enumerate(cases):')[0])
for case in cases:
    d,g = case['dataset'],case['gt']
    tr,te = d['train'],d['test']
    t = d['eval_times']; sy=d['yield_stress']; extra=(d['past_dt'],t,sy)
    def forecast(fit,target):
        xf,xv,w=m._align(fit,target)
        tau,b=m._profile(m._Response(fit,xf,*extra),fit,w)
        a,r=m._Response(target,xv,*extra).design(tau)
        return target['strain_0'][:,None]+a@b, target['instant_rate_obs']-r@b
    def subset(s, mask):
        return {k:v[mask] for k,v in s.items()}
    labels=('unchanged','full innovation','soft 0.005','soft 0.010','soft 0.015','soft 0.020')
    def adjust(delta):
        return [np.zeros_like(delta),delta,*[np.sign(delta)*np.maximum(abs(delta)-c,0) for c in (.005,.010,.015,.020)]]
    folds=[]
    for lab in np.unique(tr['lab_id']):
        mask=tr['lab_id']==lab
        f,v=subset(tr,~mask),subset(tr,mask)
        p,delta=forecast(f,v)
        baseline=p-v['strain_curves_obs']
        scores=[np.minimum(np.abs(baseline+q[:,None]*t),.25).mean() for q in adjust(delta)]
        folds.append(scores)
        print('train-only held lab',lab,'clipped curve MAE',np.round(scores,5))
    print('labels:',labels)
    print('training-only equal-lab mean / worst:', np.round(np.mean(folds,axis=0),5), np.round(np.max(folds,axis=0),5))
    def report(f,v,truth):
        p,delta=forecast(f,v)
        hot=(abs(v['stress_hold_0'])>sy)|(abs(v['past_stress'])>sy).any(axis=1)
        ls=v['lab_id']; cells=np.column_stack((ls,v['stratum']))
        for name,q in zip(labels,adjust(delta)):
            e=abs(p+q[:,None]*t-truth)
            print(name,'MAE all/hot/cold/worst cell',np.round([e.mean(),e[hot].mean(),e[~hot].mean(),max(e[(cells==c).all(axis=1)].mean() for c in np.unique(cells,axis=0))],5))
    print('full public'); report(tr,te,g['te_clean'])
    mask=te['lab_id']==2
    print('public lab 2, outcomes omitted'); report(subset(tr,tr['lab_id']!=2),subset(te,mask),g['te_clean'][mask])
