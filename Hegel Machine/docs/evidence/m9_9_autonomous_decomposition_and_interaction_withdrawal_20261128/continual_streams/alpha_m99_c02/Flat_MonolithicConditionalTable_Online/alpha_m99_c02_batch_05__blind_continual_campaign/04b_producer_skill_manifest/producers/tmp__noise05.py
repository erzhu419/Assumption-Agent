exec(open('/tmp/audit05.py').read().split('for loss,cs,rs in')[0])
v=fit(np.ones(len(lab),bool),'baseline',1,1); e,z=residual(v)
print('time-wise curve median',np.median(e,axis=0).round(5)); print('time-wise curve MAD',np.median(abs(e-np.median(e,axis=0)),axis=0).round(5)); print('trimmed curve/rate correlations',np.corrcoef(np.column_stack((np.clip(e,-.03,.03),np.clip(z,-.025,.025))).T).round(2))
for j in labs:
 q=lab==j; print('lab',j,'curve signed median',np.median(e[q],axis=0).round(4),'rate signed median',round(np.median(z[q]),5))
print('train hold/past and cal channel ranges')
for key in ('hold_channel_raw','past_channel_raw'):
 a=m['_invert_bridge'](tr[key],b); br=s['calibration_load_cells']['true_channel_loads']; print(key,a.reshape(-1,4).min(0).round(2),a.reshape(-1,4).max(0).round(2),br.min(0),br.max(0))
for cw,rw in [(1.,1.),(1.,.33),(1.,3.),(1.,10.)]:
 def weighted(q):
  vf=fit(q,'baseline',1,1)
  def f(v):
   ec,er=residual(v); return np.r_[ec[q].ravel()*cw/.012,er[q]*rw/.007]
  return least_squares(f,vf,loss='soft_l1',f_scale=1.5).x
 vv=weighted(lab==lab); vals=[]
 for j in labs:
  q=lab==j; vf=weighted(~q); ec,er=residual(vf); vals.append([*np.exp(vf[:2]),np.median(abs(ec[q])),np.median(abs(er[q]))])
 print('rate weight',rw,'full times',np.exp(vv[:2]).round(5),'folds (times, curve MAE median, rate MAE median)',np.array(vals).round(5))
