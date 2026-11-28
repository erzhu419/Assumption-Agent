import numpy as np,pickle,glob,importlib.util,copy
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];t=s['train'];x=t['x'];Y=np.c_[t['y'],t['traj']]; np.set_printoptions(precision=4,suppress=True)
print('times,tel',s['eval_times'],s['telemetry_vec']);print('ranges',x.min(0),x.max(0),Y.std(0));print('sv time',np.linalg.svd(Y[:,1:],compute_uv=False));print('y on traj residual',np.std(np.c_[np.ones(len(Y)),Y[:,1:]]@np.linalg.lstsq(np.c_[np.ones(len(Y)),Y[:,1:]],Y[:,0],rcond=1e-9)[0]-Y[:,0]));
a=copy.deepcopy(s);a['test']['x']=x[52:];a['train']={k:v[:52] if k!='summary' else v for k,v in t.items()}
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 m=importlib.util.module_from_spec(sp:=importlib.util.spec_from_file_location('m',f));sp.loader.exec_module(m)
 try:
  p=m.solve_batch(a);print(f[-18:],np.mean(abs(p['y_hat']-Y[52:,0])),np.mean(abs(p['state_pred']-Y[52:,1:])))
 except Exception as e: print(e)
for name in ('panel_a','panel_b','panel_c'):
 p=s[name];u=p['u'];print(name,'sample',u[:2]);
 for k,v in p.items():
  if k=='u':continue
  print(k,'std',v.std(0),'correlation',np.corrcoef(u.T,v.T)[:4,4:]);
  fs={'linear':u,'squares':u*u,'pairs':np.array([u[:,i]*u[:,j] for i in range(4) for j in range(i+1,4)]).T,'sin':np.sin(2*u),'cos':np.cos(2*u),'tanh':np.tanh(2*u),'linear+squares':np.c_[u,u*u],'linear+pairs':np.c_[u,np.array([u[:,i]*u[:,j] for i in range(4) for j in range(i+1,4)]).T]}
  for label,F in fs.items():
   F=np.c_[np.ones(len(u)),F];res=v-F@np.linalg.lstsq(F,v,rcond=None)[0];print(label,round(np.sqrt(np.mean(res**2)),5),end='; ')
  print()
