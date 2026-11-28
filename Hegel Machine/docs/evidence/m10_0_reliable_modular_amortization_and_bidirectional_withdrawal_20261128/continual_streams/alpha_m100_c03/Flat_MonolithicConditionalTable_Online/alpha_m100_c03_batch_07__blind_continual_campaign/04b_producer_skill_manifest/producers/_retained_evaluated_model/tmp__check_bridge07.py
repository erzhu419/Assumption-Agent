import importlib.util,pickle,numpy as np
from sklearn.model_selection import KFold
p='/tmp/hegel_continual_archive/alpha_m100_c03_batch_05_solver.py'
s=importlib.util.spec_from_file_location('arch',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; print('diagnostics',type(i.get('campaign_diagnostics')).__name__)
d=i['dataset']; c=d['calibration_load_cells']; y=c['true_channel_loads']; a=c['bridge_raw']; clean=np.max(abs(a),axis=1)<6
for mode in ('affine','opposite trigonometric'):
 es=[]; inv=[]
 for tr,va in KFold(5,shuffle=True,random_state=28).split(y):
  tr=tr[clean[tr]]; va=va[clean[va]]
  if mode=='affine':
   B=np.linalg.lstsq(np.c_[np.ones(len(tr)),y[tr]],a[tr],rcond=None)[0]; pred=np.c_[np.ones(len(va)),y[va]]@B; rec=np.linalg.solve(B[1:].T,(a[va]-B[0]).T).T
  else:
   f,b=m._bridge({'bridge_raw':a[tr],'true_channel_loads':y[tr]}); pred=(m._basis(y[va],f)*b).sum(-1); rec=m._inverse(a[va],(f,b))
  es.append(np.sqrt(np.mean((pred-a[va])**2,axis=0))); inv.append(np.sqrt(np.mean((rec-y[va]).sum(-1)**2)))
 print(mode,'worst fold/channel raw RMSE',np.max(es),'worst fold total-load RMSE',max(inv))
f,b=m._bridge(c); print('full bridge',f,b,'own min derivative',b[:,1]-abs(b[:,2]*f[0]),'cross max derivative',abs(b[:,3])+abs(b[:,4]*f[1])); sw=c['single_channel_sweeps']; z=np.zeros((len(sw['lab_id']),4)); j=sw['lab_id']; z[np.arange(len(j)),j]=sw['true_stress']; pred=(m._basis(z,f)*b).sum(-1)[np.arange(len(j)),j]; print('sweep per axis clean max residual',[np.max(abs(pred[(j==k)&(abs(sw['cmd_stress'])<6)]-sw['cmd_stress'][(j==k)&(abs(sw['cmd_stress'])<6)])) for k in range(4)])
for k in ('train','test'):
 print(k,'roundtrip and inferred load range')
 for field in ('past_channel_raw','hold_channel_raw'):
  a0=d[k][field].reshape(-1,4); x=m._inverse(a0,(f,b)); print(field,np.max(abs((m._basis(x,f)*b).sum(-1)-a0)),np.min(x,axis=0),np.max(x,axis=0))
