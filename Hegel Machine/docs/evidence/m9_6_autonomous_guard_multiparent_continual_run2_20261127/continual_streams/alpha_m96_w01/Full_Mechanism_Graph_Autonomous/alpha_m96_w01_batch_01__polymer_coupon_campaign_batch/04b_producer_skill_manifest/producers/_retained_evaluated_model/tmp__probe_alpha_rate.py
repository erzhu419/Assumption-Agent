exec(open('/tmp/check_alpha_batch.py').read().split("run('public', d)")[0])
tr, te = d['train'], d['test']
x, complete = m._features(tr, te)
n = len(tr['lab_id'])
gain,bias,_ = m._calibrate(d['calibration_load_cells'])
dt,t = d['past_dt'],d['eval_times']
def design(tau, split, x):
    j = split['lab_id']; return m._design(tau,split['past_stress_cmd']*gain[j,None]+bias[j,None],split['stress_hold_cmd']*gain[j]+bias[j],split['initial_memory_states'],dt,t,x)
j=tr['lab_id']; u=complete[:n]
tau,theta=m._fit(tr['past_stress_cmd'][u]*gain[j[u],None]+bias[j[u],None],tr['stress_hold_cmd'][u]*gain[j[u]]+bias[j[u]],tr['initial_memory_states'][u],dt,t,x[:n][u],tr['strain_curves_obs'][u]-tr['strain_0'][u,None],tr['instant_rate_obs'][u])
a,r=design(tau,te,x[n:]); p0=a@theta+te['strain_0'][:,None]; q=te['instant_rate_obs']-r@theta; e=p0-gt['te_clean']; missing=~complete[n:]
rmse=lambda e:np.sqrt(np.mean(e*e))
print('rate residual abs quantiles complete', np.quantile(abs(q[~missing]),[0,.5,.9,.95,1]))
print('rate noise against public clean reference',np.quantile(abs(te['instant_rate_obs']-gt['te_true_rate0']),[.5,.9,.95,.99,1]))
for idx in np.flatnonzero(missing): print('missing row',idx,'lab',te['lab_id'][idx],'rate innovation',round(q[idx],5),'implied needed drift',round(-e[idx,-1]/t[-1],5),'rate noise',round(te['instant_rate_obs'][idx]-gt['te_true_rate0'][idx],5))
for a in (0,.75,1.):
 p1=p0+(a*np.where(missing,q,0.))[:,None]*t
 ee=p1-gt['te_clean']; w,c=m._allocation(d,p1)
 print('correction',a,'rmse',rmse(ee),'missing',rmse(ee[missing]),'worst horizon',max(map(rmse,ee.T)),'worst lab',max(rmse(ee[te['lab_id']==j]) for j in range(4)),'allocation max',max(abs(w-gt['w_alloc_opt'])))
# Check whether the observed rate noise is likewise small in the measured training response.
a,r=design(tau,tr,x[:n]); q=tr['instant_rate_obs']-r@theta; e=tr['strain_curves_obs']-tr['strain_0'][:,None]-a@theta
s=e@t/(t@t)
print('train complete absolute rate-minus-curve-slope quantiles',np.quantile(abs(q[u]-s[u]),[.5,.9,.95,.99,1])); print('train missing same',np.quantile(abs(q[~u]-s[~u]),[.5,.9,.95,.99,1]))
