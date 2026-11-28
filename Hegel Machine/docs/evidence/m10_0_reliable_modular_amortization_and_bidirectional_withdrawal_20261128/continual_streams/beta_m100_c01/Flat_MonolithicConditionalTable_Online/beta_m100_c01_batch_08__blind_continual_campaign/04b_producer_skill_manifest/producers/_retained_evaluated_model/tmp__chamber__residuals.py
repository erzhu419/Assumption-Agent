import pickle,runpy,numpy as np
p='/tmp/hegel_continual_archive/beta_m100_c01_batch_'; m=runpy.run_path(p+'03_solver.py'); t=runpy.run_path(p+'04_solver.py')
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=it['study']; d=s['sensor_calibration_sweeps']; x=d['ref_rosette']; y=d['raw_rosette']; b=d['single_bank_sweeps']; cp=m['_fit_calibration'](x,y); print('coefs',cp); r=y-m['_forward'](x,cp); good=np.abs(y)<5; print('sweep clean rmse/max',np.sqrt(np.mean(r[good]**2)),np.max(abs(r[good]))); print('single bank residuals by bank (excluding large raw offsets):');
for j in range(4):
 g=(b['bank_id']==j)&(abs(b['raw_readout'])<5); xx=np.zeros((g.sum(),4));xx[:,j]=b['ref_signal'][g]; print(j,np.round(b['raw_readout'][g]-m['_forward'](xx,cp)[:,j],5))
S=t['_transport'](s)[0]; ij=s['sensor_ij'];
for i,pu in enumerate(it['campaign_diagnostics']):
 raw=pu['y_pulse_raw']; xp=(S@pu['w_pulse']).reshape(-1,4); rr=raw-m['_forward'](xp,cp).ravel(); print('PULSE',i,'raw residual quantiles',np.quantile(abs(rr),[0,.25,.5,.75,.9,.95,1]),'by shift',np.sqrt(np.mean(rr.reshape(3,-1)**2,axis=1))); ii=np.argsort(abs(rr))[-7:][::-1]; print('largest',list(zip(ii.tolist(),ij[ii%len(ij)].tolist(),np.round(raw[ii],5),np.round(rr[ii],5))))
