exec(open('/tmp/w03_b04_eval.py').read().split('pred=np.empty_like')[0])
def rm(a): return np.sqrt(np.mean(a*a))
out=m.solve_alpha_w03_batch_04(d); tau=out['estimated_taus']; coef=np.r_[out['hereditary_weights'],out['viscoplastic_weights']]
c,r=m._design(tau,x[:n],h[:n],z[:n],dt,t,y); ec=obs-start[:,None]-c@coef; er=rate-r@coef
for l in range(4):
 i=(lab==l)&~missing; clean=(abs(ec)<.10)&i[:,None]; print('lab',l,'slope',np.median((ec/t)[clean &(t>=1)]),'rate',np.median(er[i]),'outliers',(i[:,None]&(abs(ec)>.2)).sum()); print('time medians',np.round(np.median(ec[i],0),5))
print('asset diagnostic:'); import json
idx=json.load(open('/holdout_task/public_asset_index.json')); print(str(idx)[:1600])
# Hold out each stratum in turn: a lab intercept may remove a panel-gauge offset, but only if it transfers within each lab.
for a in np.unique(tr['stratum']):
 fit=(tr['stratum']!=a)&~missing; hold=(tr['stratum']==a)&~missing
 if not hold.any(): continue
 tau,b=m._fit(x[:n][fit],h[:n][fit],z[:n][fit],dt,t,y,obs[fit]-start[fit,None],rate[fit]); c,r=m._design(tau,x[:n],h[:n],z[:n],dt,t,y); es=obs-start[:,None]-c@b; rs=rate-r@b; corrected=es.copy(); rr=rs.copy()
 for l in range(4):
  i=fit&(lab==l); j=hold&(lab==l)
  if i.any():
   e=es[i][:,t>=1]/t[t>=1]; v=np.median(e); corrected[j]-=v*t; rr[j]-=v
 print('stratum',a,'complete n',hold.sum(),'trimmed curve/R0 base -> corrected',*(round(rm(v),6) for v in (es[hold][abs(es[hold])<.1],corrected[hold][abs(es[hold])<.1],rs[hold],rr[hold])))
