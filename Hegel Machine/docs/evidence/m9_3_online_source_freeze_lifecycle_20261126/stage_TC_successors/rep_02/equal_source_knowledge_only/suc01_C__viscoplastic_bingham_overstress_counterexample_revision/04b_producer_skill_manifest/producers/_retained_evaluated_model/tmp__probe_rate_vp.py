exec(open('/tmp/stress_vp.py').read().split('for i, case in enumerate(cases):')[0])
for case in cases:
    d, g = case['dataset'], case['gt']
    tr, te = d['train'], d['test']
    extra = (d['past_dt'], d['eval_times'], d['yield_stress'])
    print('times', extra[1])
    def audit(label, fit, target, clean=None):
        xf, xv, w = m._align(fit, target)
        tau, beta = m._profile(m._Response(fit, xf, *extra), fit, w)
        a, r = m._Response(target, xv, *extra).design(tau)
        pred = target['strain_0'][:, None] + a @ beta
        delta = target['instant_rate_obs'] - r @ beta
        t = extra[1]
        def q(x): return np.round(np.quantile(x,[0,.1,.5,.9,1]),4)
        print(label, 'rate innovations', q(delta))
        if clean is not None:
            print('rate measurement error',q(target['instant_rate_obs']-g['te_true_rate0']))
            for lab in np.unique(target['lab_id']):
                j = target['lab_id']==lab
                bias = np.median(delta[j])
                print('lab',lab,'rate median',round(bias,4),'MAE raw/lab rate shift/individual rate shift',np.round([np.abs((pred-clean)[j]).mean(),np.abs((pred-clean)[j]+bias*t).mean(),np.abs((pred-clean)[j]+delta[j,None]*t).mean()],5))
        else:
            obs=target['strain_curves_obs']
            # Median pointwise slope on positive times is insensitive to isolated curve glitches.
            j=t>0
            slope=np.median((obs[:,j]-pred[:,j])/t[j],axis=1)
            print('curve-derived slope / innovation disagreement',q(slope),q(delta-slope))
            for shift in [np.zeros_like(delta),np.full_like(delta,np.median(delta)),delta]:
                e=pred+shift[:,None]*t-obs
                print(' median absolute curve residual:',round(np.median(abs(e)),5))
    audit('public test',tr,te,g['te_clean'])
    for lab in np.unique(tr['lab_id']):
        k=tr['lab_id']==lab
        audit('training-only held lab '+str(lab),{a:b[~k] for a,b in tr.items()},{a:b[k] for a,b in tr.items()})
