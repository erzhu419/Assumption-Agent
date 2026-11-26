exec(open('/tmp/w03_b04_eval.py').read().split('pred=np.empty_like')[0])
def rm(e): return np.sqrt(np.mean(e*e))
out=m.solve_alpha_w03_batch_04(d)
for a,b in [('estimated_taus','taus'),('hereditary_weights','theta'),('viscoplastic_weights','vp_theta')]:
 print(a, 'fit',out[a], 'ref',gt[b])
miss=np.isnan(te['coupon_features_raw']).any(1); act=(abs(h[n:])>y)|(abs(z[n:]).max(1)>y)
e=out['predicted_strain']-gt['te_clean']; er=out['initial_rate_pred']-gt['te_true_rate0']; dr=te['instant_rate_obs']-out['initial_rate_pred'];
for name,j in [('all',np.ones(len(miss),bool)),('complete',~miss),('missing',miss)]+[(f'lab{l}/yield={q}/missing={a}',(te['lab_id']==l)&(act==q)&(miss==a)) for l in range(4) for q in [False,True] for a in [False,True]]:
 if j.any(): print(name,j.sum(),'curve/adjusted/early/late/rate/observed-rate',*[round(v,7) for v in (rm(e[j]),rm(e[j]+miss[j,None]*dr[j,None]*t),rm(e[j,0]),rm(e[j,-1]),rm(er[j]),rm((er+dr)[j]))])
coef=np.r_[out['hereditary_weights'],out['viscoplastic_weights']]; c,r=m._design(out['estimated_taus'],x[:n],h[:n],z[:n],dt,t,y); resid=start[:,None]+c@coef-obs
print('complete residual curve quantiles',np.quantile(abs(resid[~missing]),[.5,.9,.95,.98,.99,1])); print('missing rate obs error correlation',np.corrcoef(dr[miss],er[miss])[0,1]);
print('history by lab and episode:')
import json
for row in json.loads(m._HISTORY.read_text()):
 print(row['episode_short_name'],[None if (v:=m._sweep(*m._checks(row,l))) is None else np.round(v,5).tolist() for l in range(4)])
print('present',[np.round(v,5).tolist() for v in m._calibrate(d['calibration_load_cells'],np.arange(4)).values()])
