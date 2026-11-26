print('public truth comparison for baseline only:')
out=s.solve_alpha_w02_batch_04(d)
for k,g in zip(out,('taus','theta','vp_theta','te_clean','te_true_rate0')):
    pred,tgt=np.asarray(out[k]),np.asarray(case['gt'][g]); print(k,pred.shape,tgt.shape,'RMSE',np.sqrt(np.mean((pred-tgt)**2)),'max error',np.max(abs(pred-tgt)))
